import sys
from glob import glob
from PIL import Image
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
import albumentations as A

sys.path.append('../')
from settings import CFG



class Img_dataset(Dataset):
    def __init__(self, paths, train_tran, val_tran, albu=True):
        super().__init__()
        self.paths = paths 
        self.train_tran = train_tran
        self.val_tran = val_tran
        self.albu = albu
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        file_path = self.paths[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.albu:
            image = self._get_aug()(image=image)['image']

        return (self.train_tran(image=image)['image'], 
                self.val_tran(image=image)['image'])
    
    def _get_aug(self):
        transform = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
#            A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.5),
            ]
        return A.Compose(transform)


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


def get_dataloader(query, batch_size, shuffle=True, albu=True):
    paths = sorted(list(glob(query)))
    dataset = Img_dataset(paths, train_tran=get_transform(TRAIN=False),
                                val_tran=get_transform(TRAIN=True),
                                albu=albu)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return paths, data_loader
            

