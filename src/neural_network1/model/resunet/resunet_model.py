""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .resunet_parts import *


class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = nn.Sequential(
            ResBlock(in_channels, 32),
            ResBlock(32, 32),
        )
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.latent = DilatedResBlock(512, 512, mid_channels=1024)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.outc = nn.Sequential()
        if out_channels != 32:
            self.outc = nn.Sequential(
                ResBlock(32, out_channels),
                # OutConv(32, out_channels)
            )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.latent(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
