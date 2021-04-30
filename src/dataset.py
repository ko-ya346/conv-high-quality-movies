from PIL import Image
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
import albumentations as A


from .config import CFG



class Img_dataset(Dataset):
    def __init__(self, paths, train_tran, val_tran):
        super().__init__()
        self.paths = paths
        self.train_tran = train_tran
        self.val_tran = val_tran
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        file_path = self.paths[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        return (self.train_tran(image=image)['image'], 
                self.val_tran(image=image)['image'])


def get_transform(*, TRAIN=False):
    if TRAIN:
        return A.Compose([
            A.Resize(CFG.low_xsize, CFG.low_ysize),
            A.Resize(CFG.xsize, CFG.ysize, interpolation=Image.NEAREST),
            ToTensorV2()
            
        ])
    else:
        return A.Compose([
            A.Resize(CFG.xsize, CFG.ysize, interpolation=Image.NEAREST),
            ToTensorV2(),
    ])
