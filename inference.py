import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

sys.path.append('./')

from settings import CFG
from src.dataset import Img_dataset, get_transform
from src.model import Generator
from src.utils import send_line_message



def inference():
    paths = sorted([f'{CFG.IMG_DIR}/{path}' for path in os.listdir(CFG.IMG_DIR)])
    train_dataset = Img_dataset(paths, train_tran=get_transform(TRAIN=False),
                            val_tran=get_transform(TRAIN=True), albu=False)
    save_name = [path.split('/')[-1] for path in paths]

    model_G = Generator()
    model_G = nn.DataParallel(model_G)
    model_G = model_G.to(device)

    states = torch.load(CFG.trained_param)
    model_G.load_state_dict(states)
    train_loader = DataLoader(train_dataset, batch_size=1)
    if not os.path.exists(CFG.OUTPUT_IMG):
        os.makedirs(CFG.OUTPUT_IMG)

    for i, (image, _) in tqdm(enumerate(train_loader)):
        image = image.to(device)
        gen_image = model_G(image)
        save_image(gen_image, f'{CFG.OUTPUT_IMG}/{save_name[i]}')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    torch.backends.cudnn.benchmark = True
    inference()
    
    text = 'finish inference'
    send_line_message(CFG.LINE_TOKEN, text)
