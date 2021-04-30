import cv2
import matplotlib.pyplot as plt


def show_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)    
    
    plt.imshow(img)
    plt.show()


def conv_scale(arr):
    ma = arr.max()
    mi = arr.min()
    return (arr-mi) / (ma-mi)
