import os
import sys
from glob import glob

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
from dotenv import load_dotenv


sys.path.append('../')
sys.path.append('./')

from settings import CFG
from src.dataset import get_dataloader, get_transform
from src.model import get_network 
from src.utils import get_line_token, send_line_message


dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

line_token = get_line_token(os.getenv('LINE_TOKEN_PATH'))

def train():
    dataset_queries = {
            '4-3': [
                os.path.join(os.getenv('INPUT_DIR'), 
                                f'4-3/images/{CFG.dataset}/*.png'),
                os.path.join(os.getenv('INPUT_DIR'),
                    f'4-3/images/{CFG.valid}/image-00*.png')
                ],
            '16-9': [
                os.path.join(os.getenv('INPUT_DIR'), 
                    f'16-9/images/{CFG.dataset}/*.png'),
                os.path.join(os.getenv('INPUT_DIR'),
                    f'16-9/images/{CFG.valid}/*.png')
                ],
            }

    _, train_loader = get_dataloader(dataset_queries.get(CFG.aspect_ratio)[0], 
        CFG.batch_size)
    _, valid_loader = get_dataloader(dataset_queries.get(CFG.aspect_ratio)[1], 
        CFG.batch_size)

    model = get_network(CFG.model)()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                lr=0.01, weight_decay=1e-8)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
    if os.path.exists(f'os.getenv("OUTPUT_DIR")/{CFG.model}'):
        states = torch.load(f'os.getenv("OUTPUT_DIR")/model/{CFG.model}_010.pytorch')
        model.load_state_dict(states)
        
    # 損失関数
    criterion = nn.MSELoss()

    # エラー推移
    metrics = {}
    metrics["train"] = []
    metrics["valid"] = []
    save_interval =5 
    max_epoch = CFG.epoch

    for epoch in tqdm(range(max_epoch)):
        print(f'epoch: {epoch+1}')
        scheduler.step()
        model.train()
        
        epoch_loss = []
        
        for num, (label, img_input) in enumerate(train_loader):
            batch_len = len(label)
            label /= 255
            img_input /= 255
            label, img_input = label.to(device), img_input.to(device)

            output = model(img_input)
            output_tensor = output.detach()

            loss = criterion(output, label)

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
            torch.save(model.state_dict(), f"{output_model}/{CFG.model}_{epoch:03}.pytorch")     

        # 生成画像を保存
        save_image(output_tensor[:10], 
                os.path.join(output_model, f'train_{epoch:02}_gen.png'))
        save_image(label[:10], 
                os.path.join(output_model, f'train_{epoch:02}_label.png'))



        # valid
        model.eval()
        for label, img_input in valid_loader:
            valid_loss = []

            label /= 255
            img_input /= 255
            label, img_input = label.to(device), img_input.to(device)

            output = model(img_input)
            loss = criterion(output, label)
            output_tensor = output.detach()
            valid_loss.append(loss.item())
        metrics['valid'].append(np.mean(valid_loss)) 

        print(f"train: {metrics['train'][-1]}")
        print(f"valid: {metrics['valid'][-1]}")
        
        # 生成画像を保存
        save_image(output_tensor[:10], 
                os.path.join(output_model, f'valid_{epoch:02}_gen.png'))
        save_image(label[:10], 
                os.path.join(output_model, f'valid_{epoch:02}_label.png'))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    torch.backends.cudnn.benchmark = True # autotunerが高速化
    train()

    text = 'finish train!!'
    send_line_message(line_token, text)
