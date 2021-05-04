import os
import sys
from glob import glob
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR
import torchvision
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

def inference():
    dataset_queries = {
            '4-3': [
                os.path.join(os.getenv('INPUT_DIR'), 
                                f'images/{CFG.dataset}/*.png'),
                os.path.join(os.getenv('INPUT_DIR'),
                    f'images/{CFG.valid}/*.png')
                ],
            }

    paths, valid_loader = get_dataloader(dataset_queries.get(CFG.aspect_ratio)[1], 
        CFG.batch_size,
        shuffle=False,
        albu=False
        )
    save_name = [path.split('/')[-1] for path in paths]

    model = get_network(CFG.model)()
    model = model.to(device)

    states = torch.load(f'{os.getenv("OUTPUT_DIR")}/model/{CFG.model}_000.pytorch')
    model.load_state_dict(states)
    
    output_dir = os.path.join(os.getenv('OUTPUT_DIR'), 'images', CFG.valid)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i, (image, _) in enumerate(tqdm(valid_loader)):
        image /= 255
        image = image.to(device)
        gen_image = model(image)
        gen_image_tensor = gen_image.detach()
        save_image(gen_image_tensor, f'{output_dir}/{save_name[i]}')
        

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    torch.backends.cudnn.benchmark = True # autotunerが高速化
    inference()

    text = 'finish train!!'
    send_line_message(line_token, text)

