import torch
from torch import nn

class AbstractNet(nn.Module):
    only_luminance = False
    input_upscale = True


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.conv_bn_relu(3, 32, kernel_size=5) # 32x96x96
        self.enc2 = self.conv_bn_relu(32, 64, kernel_size=3, pool_kernel=3)  # 64x24x24
        self.enc3 = self.conv_bn_relu(64, 128, kernel_size=3, pool_kernel=2)  # 128x12x12
        self.enc4 = self.conv_bn_relu(128, 256, kernel_size=3, pool_kernel=2)  # 256x6x6
        
        self.dec1 = self.conv_bn_relu(256, 128, kernel_size=3, pool_kernel=-2)  # 128x12x12
        self.dec2 = self.conv_bn_relu(128 + 128, 64, kernel_size=3, pool_kernel=-2)  # 64x24x24
        self.dec3 = self.conv_bn_relu(64 + 64, 32, kernel_size=3, pool_kernel=-3)  # 32x96x96
        self.dec4 = nn.Sequential(
            nn.Conv2d(32 + 32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, pool_kernel=None):
        layers = []
        if pool_kernel is not None:
            if pool_kernel > 0:
                layers.append(nn.AvgPool2d(pool_kernel))
            elif pool_kernel < 0:
                layers.append(nn.UpsamplingNearest2d(scale_factor=-pool_kernel))
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, padding=(kernel_size - 1) // 2))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        out = self.dec1(x4)
        out = self.dec2(torch.cat([out, x3], dim=1))
        out = self.dec3(torch.cat([out, x2], dim=1))
        out = self.dec4(torch.cat([out, x1], dim=1))
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.conv_bn_relu(6, 16, kernel_size=5, reps=1) # fake/true color + gray
        self.conv2 = self.conv_bn_relu(16, 32, pool_kernel=4)
        self.conv3 = self.conv_bn_relu(32, 64, pool_kernel=2)
        self.conv4 = self.conv_bn_relu(64, 128, pool_kernel=2)
        self.conv5 = self.conv_bn_relu(128, 256, pool_kernel=2)
        self.out_patch = nn.Conv2d(256, 1, kernel_size=1) #1x3x3

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, pool_kernel=None, reps=2):
        layers = []
        for i in range(reps):
            if i == 0 and pool_kernel is not None:
                layers.append(nn.AvgPool2d(pool_kernel))
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch,
                                    out_ch, kernel_size, padding=(kernel_size - 1) // 2))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        return self.out_patch(out)

class SRCNN(AbstractNet):
    """
    SRCNN
    ref: http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html
    """

    def __init__(self, **kwargs):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.activate(self.conv1(x))
        h = self.activate(self.conv2(h))
        return self.conv3(h)


MODELS = {
        'srcnn' : SRCNN,
        'pix2pix': Generator,
        }

def get_network(name):
    for n, cls in MODELS.itens():
        if n == name:
            return cls
    raise ValueError(f'Model {name} is not defined.')
