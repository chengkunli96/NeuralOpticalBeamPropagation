from ..dbpnet.dbpnet_model import DBPNet
from ..resunet.resunet_model import ResUNet
from .model_parts import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(Net, self).__init__()
        self.in_channels = {'img': in_channels1, 'angle': in_channels2}
        self.out_channels = out_channels

        self.net1 = ResUNet(self.in_channels['img'])
        self.middle = DownResBlock(32, 32)
        self.net2 = DBPNet(
            num_channels=32, out_channels=out_channels,
            base_filter=64, feat=256, num_stages=7,
            scale_factor=2,
        )

    def forward(self, img, angle):
        # img map
        x = self.net1(img, angle)
        x = self.middle(x)
        x = self.net2(x)
        return x
