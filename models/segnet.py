from __future__ import annotations

import torch
import torch.nn as nn


def conv_bn_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SegNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1) -> None:
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.enc1 = nn.Sequential(
            conv_bn_relu(in_channels, 64),
            conv_bn_relu(64, 64),
        )
        self.enc2 = nn.Sequential(
            conv_bn_relu(64, 128),
            conv_bn_relu(128, 128),
        )
        self.enc3 = nn.Sequential(
            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
            conv_bn_relu(256, 256),
        )
        self.enc4 = nn.Sequential(
            conv_bn_relu(256, 512),
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
        )
        self.enc5 = nn.Sequential(
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
        )

        self.dec5 = nn.Sequential(
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
        )
        self.dec4 = nn.Sequential(
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 256),
        )
        self.dec3 = nn.Sequential(
            conv_bn_relu(256, 256),
            conv_bn_relu(256, 256),
            conv_bn_relu(256, 128),
        )
        self.dec2 = nn.Sequential(
            conv_bn_relu(128, 128),
            conv_bn_relu(128, 64),
        )
        self.dec1 = nn.Sequential(
            conv_bn_relu(64, 64),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        size1 = x1.size()
        x, idx1 = self.pool(x1)

        x2 = self.enc2(x)
        size2 = x2.size()
        x, idx2 = self.pool(x2)

        x3 = self.enc3(x)
        size3 = x3.size()
        x, idx3 = self.pool(x3)

        x4 = self.enc4(x)
        size4 = x4.size()
        x, idx4 = self.pool(x4)

        x5 = self.enc5(x)
        size5 = x5.size()
        x, idx5 = self.pool(x5)

        x = self.unpool(x, idx5, output_size=size5)
        x = self.dec5(x)

        x = self.unpool(x, idx4, output_size=size4)
        x = self.dec4(x)

        x = self.unpool(x, idx3, output_size=size3)
        x = self.dec3(x)

        x = self.unpool(x, idx2, output_size=size2)
        x = self.dec2(x)

        x = self.unpool(x, idx1, output_size=size1)
        x = self.dec1(x)
        return x
