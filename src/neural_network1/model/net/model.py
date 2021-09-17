from ..dbpnet.dbpnet_model import DBPNet
from ..resunet.resunet_model import ResUNet
from  .model_parts import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, in_channels, out_channels, middle_channels=32):
        super(Net, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ResUNet part
        self.net1 = ResUNet(in_channels, middle_channels)

        # middle supervising part
        self.middle1 = nn.Sequential(
            nn.Conv2d(middle_channels, middle_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
        )
        self.middle2 = nn.Conv2d(middle_channels, middle_channels, kernel_size=1, padding=0, bias=False)
        self.middle3 = nn.Conv2d(middle_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.middle4 = nn.Conv2d(out_channels, middle_channels, kernel_size=1, padding=0, bias=False)

        # DBPNet part
        self.down = DownResBlock(middle_channels, middle_channels)
        self.net2 = DBPNet(
            in_channels=middle_channels, out_channels=out_channels,
            base_filter=64,  feat=256, num_stages=7,
            scale_factor=2,
        )

    def forward(self, input):
        # ResUnet
        x = self.net1(input)

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
