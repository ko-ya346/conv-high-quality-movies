import cv2
import matplotlib.pyplot as plt
import requests
import json


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
    return (arr-mi) / (ma-mi)


def send_line_message(token, text):
    '''
    LINEにtextの内容を送信する。
    '''
    line_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {token}'}
    data = {'message': f'message: {text:}'}
    requests.post(line_api, headers=headers, data=data)


def get_line_token(token_path):
    with open(token_path, 'r') as f:
         token = json.load(f)['token']
    return token


def resize_img(image, magnification=3):
    '''
    cv2でリサイズを行う
    '''
    return cv2.resize(image, (image.shape[1]*magnification,
                              image.shape[0]*magnification),
                      interpolation=cv2.INTER_CUBIC)
