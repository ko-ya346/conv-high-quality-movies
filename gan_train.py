import os

import numpy as np
import torch
from config import CFG
from dotenv import load_dotenv
from src.dataset import TrainDataset
from src.lossess import GANLoss
from src.model import Discriminator, Generator
from src.utils import get_line_token, send_line_message
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

line_token = get_line_token(os.getenv("LINE_TOKEN_PATH"))


def train(debug=False):
    max_epoch = CFG.epoch
    if debug:
        max_epoch = 1

    dataset_queries = {
        "T91": os.path.join(os.getenv("INPUT_DIR"), "T91/*.png"),
        "Set5": os.path.join(os.getenv("INPUT_DIR"), "Set5/*.png"),
        "Set14": os.path.join(os.getenv("INPUT_DIR"), "Set14/*.png"),
    }

    # train dataloader
    train_query = dataset_queries.get(CFG.train_dataset)
    train_dataset = TrainDataset(
        train_query,
        max_size=CFG.max_size,
        shrink_scale=CFG.shrink_scale,
        total_samples=CFG.total_samples,
        input_upsample=CFG.input_upsample,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
    )

    # valid dataloader
    valid_query = dataset_queries.get(CFG.valid_dataset)
    valid_dataset = TrainDataset(
        valid_query,
        max_size=CFG.max_size,
        shrink_scale=CFG.shrink_scale,
        total_samples=CFG.total_samples,
        input_upsample=CFG.input_upsample,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
    )

    # model
    model_G = nn.DataParallel(Generator()).to(device)
    model_D = nn.DataParallel(Discriminator()).to(device)

    # optimizer
    optimizer_G = torch.optim.Adam(
        model_G.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    optimizer_D = torch.optim.Adam(
        model_D.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    scheduler_G = StepLR(optimizer_G, step_size=CFG.lr_step, gamma=CFG.lr_gamma)
    scheduler_D = StepLR(optimizer_D, step_size=CFG.lr_step, gamma=CFG.lr_gamma)

    # 損失関数
    # なにこれ
    criterionGAN = GANLoss().to(device)
    mae_loss = nn.L1Loss()

    # エラー推移
    metrics = {}
    metrics["generator"] = []
    metrics["discriminator"] = []
    metrics["valid"] = []
    save_interval = 5

    # train loop
    for epoch in tqdm(range(max_epoch)):
        print(f"epoch: {epoch+1}")
        scheduler_G.step()
        scheduler_D.step()
        model_G.train()
        model_D.train()

        epoch_loss_G = []
        epoch_loss_D = []

        for num, (img_input, img_real) in enumerate(train_loader):
            img_input, img_real = img_input.to(device), img_real.to(device)

            # 偽画像
            img_fake = model_G(img_input)
            img_fake_tensor = img_fake.detach()

            LAMBD = 1.0  # BCEとMAEの係数

            #            print('img_fake:', img_fake.size())
            #            print('img_real:', img_real.size())

            output_D = model_D(torch.cat([img_fake, img_real], dim=1))
            #            print(f'{output_D[0]}')
            #            print(f'output_D.size(): {output_D.size()}')

            # loss計算
            # generatorは本物と判別されてほしい
            loss_G_bce = criterionGAN(output_D, True)
            loss_G_mae = LAMBD * mae_loss(img_fake, img_real)
            loss_G_sum = loss_G_bce + loss_G_mae

            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            loss_G_sum.backward()
            optimizer_G.step()

            # Discriminatoの訓練
            real_out = model_D(torch.cat([img_real, img_input], dim=1))
            loss_D_real = criterionGAN(real_out, True)

            # 偽の画像をニセと識別できるようにする
            #            print('img_input:', img_input.size())

            # 20210618
            # Discriminatorのbackwardがうまくいかなかった原因
            # img_fake -> img_fake_tensor に変えたらうまくいった
            fake_out = model_D(torch.cat([img_fake_tensor, img_input], dim=1))

            #            print(f'fake_out.size(): {fake_out.size()}')
            #            print(fake_out)

            loss_D_fake = criterionGAN(fake_out, False)
            #            print(loss_D_fake)
            #            print(loss_D_real)

            loss_D_sum = loss_D_real + loss_D_fake

            optimizer_G.zero_grad()
            optimizer_D.zero_grad()
            loss_D_sum.backward()

            optimizer_D.step()

            epoch_loss_G.append(loss_G_sum.item())
            epoch_loss_D.append(loss_D_sum.item())

        metrics["generator"].append(np.mean(epoch_loss_G))
        metrics["discriminator"].append(np.mean(epoch_loss_D))

        # モデルの保存
        output_model = f"{os.getenv('OUTPUT_DIR')}/model"
        if not os.path.exists(output_model):
            os.mkdir(output_model)
        if epoch % save_interval == 0 or epoch == max_epoch:
            torch.save(
                model_G.state_dict(), f"{output_model}/gen_{epoch:03}.pytorch"
            )
            torch.save(
                model_D.state_dict(), f"{output_model}/dis_{epoch:03}.pytorch"
            )

        # 画像を保存

        # 低画質→高画質に変換した画像
        save_image(
            img_fake_tensor[:10],
            os.path.join(output_model, f"train_{epoch:02}_gen.png"),
        )
        # 元画像
        save_image(
            img_real[:10],
            os.path.join(output_model, f"train_{epoch:02}_target.png"),
        )

        # valid
        model_G.eval()
        for img_input, img_real in valid_loader:
            valid_loss = []

            img_input, img_real = img_input.to(device), img_real.to(device)

            output = model_G(img_input)
            loss = mae_loss(output, img_real)
            output_tensor = output.detach()
            valid_loss.append(loss.item())
        metrics["valid"].append(np.mean(valid_loss))

        print(f"generator: {metrics['generator'][-1]}")
        print(f"discriminator: {metrics['discriminator'][-1]}")
        print(f"valid: {metrics['valid'][-1]}")

        # 生成画像を保存
        save_image(
            output_tensor[:10],
            os.path.join(output_model, f"valid_{epoch:02}_gen.png"),
        )
        save_image(
            img_real[:10],
            os.path.join(output_model, f"valid_{epoch:02}_target.png"),
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.backends.cudnn.benchmark = True  # autotunerが高速化
    train()

    text = "finish train!!"
    send_line_message(line_token, text)
