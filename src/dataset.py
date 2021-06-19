from glob import glob

from PIL import Image
from src.utils import calculate_original_img_size
from torch.utils.data import Dataset
from torchvision import transforms as T


class TrainDataset(Dataset):
    """
    Glob Query にマッチする画像からランダムに切り取られた正方形の画像と
    それを縮小した画像のペアを返すデータセット
    引数によって既に縮小された画像を返すかどうかが変わってくるので注意
    Args:
        query: glob query
        max_size:
            ランダムに切り取る画像の最大 pixel 数.
            正確な値は shrink_scale とともに
            `calculate_original_img_size` によって計算される値になります.
        shrink_scale:
            切り取られた画像サイズを変換するスケール.
            例えば 3 が設定されると, 切り取られた画像を 1/3 に縮小します.
        total_samples:
            1 epoch あたりの画像枚数.
            指定しないとすべての画像を一通り見ると 1epoch になります.
            特定の値を設定すると, その数だけランダムにデータを
            pickup したときに 1epoch とします.
        input_upsample:
            True のとき入力画像を resize して target 画像と同じ大きさに前もって変換します.
            変換方法は `interpolation` で指定されたアルゴリズムを使用します.
        interpolation:
    """

    def __init__(
        self,
        query,
        max_size=128,
        shrink_scale=3,
        total_samples=10000,
        input_upsample=True,
        interpolation=Image.BICUBIC,
    ):
        super().__init__()

        self.img_paths = list(glob(query))

        if total_samples:
            self.total_samples = total_samples
        else:
            self.total_samples = len(self.img_paths)

        self.n_images = len(self.img_paths)

        high_size = calculate_original_img_size(max_size, upscale_factor=shrink_scale)
        low_size = int(high_size / shrink_scale)

        if input_upsample:
            self.input_transform = T.Compose(
                [
                    T.Resize(size=low_size),
                    T.Resize(size=high_size, interpolation=interpolation),
                    T.ToTensor(),
                ]
            )
        else:
            self.input_transform = T.Compose(
                [T.Resize(size=low_size, interpolation=interpolation), T.ToTensor()]
            )

        self.target_transform = T.Compose([T.ToTensor()])

        self.preprocess = T.Compose(
            [T.RandomCrop(size=high_size), T.RandomHorizontalFlip()]
        )

        # max_sizeより画像が小さい場合は拡大する
        self.upscale_transform = T.Compose(
            [
                T.Resize(size=max_size, interpolation=interpolation),
            ]
        )

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        file_path = self.img_paths[idx % self.n_images]
        img = Image.open(file_path)

        try:
            img = self.preprocess(img)
        except:
            img = self.upscale_transform(img)
            img = self.preprocess(img)

        img_target = img.copy()
        img_target = self.target_transform(img_target)

        img_input = self.input_transform(img)

        return img_input, img_target


class TestDataset(Dataset):
    """
    Args:
        query:
            glob query
        xsize:
            入力画像のx軸方向のサイズ
        ysize:
            入力画像のy軸方向のサイズ
        up_scale:
            画像を拡大するサイズ
        interpolation:
    """

    def __init__(self, query, xsize, ysize, up_scale=3, interpolation=Image.BICUBIC):
        super().__init__()

        self.img_paths = list(glob(query))
        self.len_img = len(self.img_paths)

        high_xsize = int(xsize * up_scale)
        high_ysize = int(ysize * up_scale)

        self.input_transform = T.Compose(
            [
                T.Resize(size=(high_xsize, high_ysize), interpolation=interpolation),
                T.ToTensor(),
            ]
        )

        self.target_transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return self.len_img

    def __getitem__(self, idx):
        file_path = self.img_paths[idx]
        img = Image.open(file_path)

        img_input = self.input_transform(img)

        return img_input
