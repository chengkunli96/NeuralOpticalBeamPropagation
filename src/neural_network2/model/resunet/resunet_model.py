""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .resunet_parts import *


class ResUNet(nn.Module):
    def __init__(self, in_channels):
        super(ResUNet, self).__init__()
        self.in_channels = in_channels
        # self.out_channels = out_channels

        self.inc = nn.Sequential(
            ResBlock(in_channels, 32),
            ResBlock(32, 32),
        )
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.latent = DilatedResBlock(512, 448, mid_channels=1024)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        # self.outc = nn.Sequential(
        #     ResBlock(32, 32),
        #     OutConv(32, out_channels)
        # )
        self.angle_model1 = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 256),
            nn.ReLU(inplace=True),
        )
        self.angle_model2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, angle):
        x1 = self.inc(x)
        # print(x1.size())
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.latent(x5)

        y = self.angle_model1(angle)  # BC
        y = y.view(y.size()[0], 1, 16, 16)
        y = self.angle_model2(y)

        x5 = torch.cat([x5, y], dim=1)


        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)


        # x = self.outc(x)
        return x
