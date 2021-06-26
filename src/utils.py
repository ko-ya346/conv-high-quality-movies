import json
import logging
import os
import random
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorboardX as tbx
import torch


def calculate_original_img_size(origin_size: int, upscale_factor: int) -> int:
    """
    元の画像サイズを縮小拡大したいときに元の画像をどの大きさに
    resize する必要があるかを返す関数
    例えば 202 px の画像を 1/3 に縮小することは出来ない(i.e. 3の倍数ではない)ので
    事前に 201 px に縮小しておく必要がありこの関数はその計算を行う
    すなわち
    calculate_original_img_size(202, 3) -> 201
    となる

    Args:
        origin_size:
        upscale_factor:
    Returns:
    """
    return origin_size - (origin_size % upscale_factor)


def show_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)

    plt.imshow(img)
    plt.show()


def conv_scale01(arr):
    ma = arr.max()
    mi = arr.min()
    return (arr - mi) / (ma - mi)


def send_line_message(token, text):
    """
    LINEにtextの内容を送信する。
    """
    line_api = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": f"message: {text:}"}
    requests.post(line_api, headers=headers, data=data)


def get_line_token(token_path):
    with open(token_path, "r") as f:
        token = json.load(f)["token"]
    return token


def resize_img(image, magnification=3):
    """
    cv2でリサイズを行う
    """
    return cv2.resize(
        image,
        (image.shape[1] * magnification, image.shape[0] * magnification),
        interpolation=cv2.INTER_CUBIC,
    )


def init_logger(log_file="train.log"):
    from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


class TbxSummary:
    def __init__(self, path_json, name="data/loss"):
        self.path_json = path_json
        self.name = name
        self.writer = tbx.SummaryWriter()

    def add_scalar(self, item, iteration):
        self.writer.add_scalars(self.name, item, iteration)

    def save_json(self):
        self.writer.export_scalars_to_json(self.path_json)
        self.writer.close()
