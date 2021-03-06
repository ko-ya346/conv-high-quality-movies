import os
from glob import glob

import torch

# 自作
from config import CFG
from dotenv import load_dotenv
from src.dataset import TestDataset
from src.model import Generator, get_network
from src.utils import get_line_token, send_line_message
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

line_token = get_line_token(os.getenv("LINE_TOKEN_PATH"))


def inference():
    query = os.path.join(
        os.getenv("INPUT_DIR"), f"images/{CFG.inference_dataset}/*.png"
    )

    # inference dataloader
    inference_dataset = TestDataset(
        query,
        xsize=CFG.xsize,
        ysize=CFG.ysize,
        up_scale=CFG.up_scale,
    )
    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=1,
        shuffle=False,
    )
    save_name = [path.split("/")[-1] for path in list(glob(query))]

    # model = get_network(CFG.model)()
    model = nn.DataParallel(Generator()).to(device)

    states = torch.load(
        f'{os.getenv("OUTPUT_DIR")}/{CFG.name_experiment}/gen_295.pytorch'
    )
    model.load_state_dict(states)

    output_dir = os.path.join(
        os.getenv("OUTPUT_DIR"), "images", CFG.inference_dataset
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, image in enumerate(tqdm(inference_dataloader)):
        image = image.to(device)
        gen_image = model(image)
        gen_image_tensor = gen_image.detach()
        save_image(gen_image_tensor, f"{output_dir}/{save_name[i]}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.backends.cudnn.benchmark = True  # autotunerが高速化
    inference()

    text = "finish inference!!"
    send_line_message(line_token, text)
