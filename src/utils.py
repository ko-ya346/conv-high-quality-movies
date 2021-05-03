import cv2
import matplotlib.pyplot as plt
import requests
import json


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
