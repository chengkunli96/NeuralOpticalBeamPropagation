from ..dbpnet.dbpnet_model import DBPNet
from ..resunet.resunet_model import ResUNet
from .model_parts import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, middle_channels=32):
        super(Net, self).__init__()
        self.in_channels = {'img': in_channels1, 'angle': in_channels2}
        self.out_channels = out_channels

        self.net1 = ResUNet(self.in_channels['img'])

        # middle supervising part
        self.middle1 = nn.Sequential(
            nn.Conv2d(middle_channels, middle_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
        )
        self.middle2 = nn.Conv2d(middle_channels, middle_channels, kernel_size=1, padding=0, bias=False)
        self.middle3 = nn.Conv2d(middle_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.middle4 = nn.Conv2d(out_channels, middle_channels, kernel_size=1, padding=0, bias=False)

        self.down = DownResBlock(middle_channels, middle_channels)
        self.net2 = DBPNet(
            num_channels=32, out_channels=out_channels,
            base_filter=64, feat=256, num_stages=7,
            scale_factor=2,
        )

    def forward(self, img, angle):
        # UNet Part
        x = self.net1(img, angle)

        # Middle Part
        x = self.middle1(x)
        x1 = self.middle2(x)
        supervisor = self.middle3(x)
        x2 = self.middle4(supervisor)
        x = x1 + x2

        # DBPNet
        x = self.down(x)
        x = self.net2(x)
        return x, supervisor
