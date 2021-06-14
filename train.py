import os
import sys
from glob import glob

import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
from dotenv import load_dotenv


sys.path.append('../')

from config import CFG
from src.dataset import TrainDataset
from src.model import get_network 
from src.utils import get_line_token, send_line_message


dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

line_token = get_line_token(os.getenv('LINE_TOKEN_PATH'))

def train(debug=False):
    max_epoch = CFG.epoch
    if debug:
        max_epoch = 1

    dataset_queries = {
            'T91':  os.path.join(os.getenv('INPUT_DIR'), 'T91/*.png'),
            'Set5':  os.path.join(os.getenv('INPUT_DIR'), 'Set5/*.png'),
            'Set14':  os.path.join(os.getenv('INPUT_DIR'), 'Set14/*.png'),
            }

    
    # train dataloader
    train_query = dataset_queries.get(CFG.train_dataset)
    train_dataset = TrainDataset(
            train_query, max_size=CFG.max_size, 
            shrink_scale=CFG.shrink_scale, 
            total_samples=CFG.total_samples, 
            input_upsample=CFG.input_upsample,
            )
    train_loader = DataLoader(
            train_dataset, batch_size=CFG.batch_size, 
            shuffle=True,
            )

    # valid dataloader
    valid_query = dataset_queries.get(CFG.valid_dataset)
    valid_dataset = TrainDataset(
            valid_query, max_size=CFG.max_size, 
            shrink_scale=CFG.shrink_scale, 
            total_samples=CFG.total_samples, 
            input_upsample=CFG.input_upsample,
            )
    valid_loader = DataLoader(
            valid_dataset, batch_size=CFG.batch_size, 
            shuffle=True,
            )

    # model
    model = get_network(CFG.model)()
    model = model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=CFG.lr, 
            weight_decay=CFG.weight_decay
            )
    scheduler = StepLR(
            optimizer, step_size=CFG.lr_step, gamma=CFG.lr_gamma
            )
    if os.path.exists(f'os.getenv("OUTPUT_DIR")/{CFG.model}'):
        states = torch.load(f'os.getenv("OUTPUT_DIR")/model/{CFG.model}_010.pytorch')
        model.load_state_dict(states)
        
    # 損失関数
    criterion = nn.MSELoss()

    # エラー推移
    metrics = {}
    metrics["train"] = []
    metrics["valid"] = []
    save_interval = 5 

    # train loop
    for epoch in tqdm(range(max_epoch)):
        print(f'epoch: {epoch+1}')
        scheduler.step()
        model.train()
        
        epoch_loss = []
        
        for num, (img_input, img_target) in enumerate(train_loader):
            img_input, img_target = img_input.to(device), img_target.to(device)

            output = model(img_input)
            output_tensor = output.detach()

            loss = criterion(output, img_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss.append(loss.item())
            
        metrics["train"].append(np.mean(epoch_loss))

        # モデルの保存
        output_model = f"{os.getenv('OUTPUT_DIR')}/model"     
        if not os.path.exists(output_model):
            os.mkdir(output_model)
        if epoch % save_interval == 0 or epoch == max_epoch:
            torch.save(
                    model.state_dict(), 
                    f"{output_model}/{CFG.model}_{epoch:03}.pytorch"
                    )

        ### 画像を保存
        # 低画質→高画質に変換した画像
        save_image(
                output_tensor[:10], 
                os.path.join(output_model, f'train_{epoch:02}_gen.png')
                )
        # 元画像
        save_image(
                img_target[:10], 
                os.path.join(output_model, f'train_{epoch:02}_target.png')
                )


        # valid
        model.eval()
        for img_input, img_target in valid_loader:
            valid_loss = []

            img_input, img_target = img_input.to(device), img_target.to(device)

            output = model(img_input)
            loss = criterion(output, img_target)
            output_tensor = output.detach()
            valid_loss.append(loss.item())
        metrics['valid'].append(np.mean(valid_loss)) 

        print(f"train: {metrics['train'][-1]}")
        print(f"valid: {metrics['valid'][-1]}")
        
        # 生成画像を保存
        save_image(
                output_tensor[:10], 
                os.path.join(output_model, f'valid_{epoch:02}_gen.png')
                )
        save_image(
                img_target[:10], 
                os.path.join(output_model, f'valid_{epoch:02}_target.png')
                )


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    torch.backends.cudnn.benchmark = True # autotunerが高速化
    train()

    text = 'finish train!!'
    send_line_message(line_token, text)
