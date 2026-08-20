from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LegacyDoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LegacyDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = LegacyDoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class LegacyUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = LegacyDoubleConv((in_channels // 2) + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(
            x,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 8,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        norm_groups = min(groups, out_channels)
        while out_channels % norm_groups != 0:
            norm_groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(norm_groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels)
        self.conv2 = ConvNormAct(out_channels, out_channels)
        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.dropout(x)
        return x + residual


class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_proj = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False)
        self.skip_proj = nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        attn = self.psi(self.gate_proj(gate) + self.skip_proj(skip))
        return skip * attn


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = ResidualSEBlock(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvNormAct(in_channels, out_channels, kernel_size=1),
        )
        self.attn = AttentionGate(out_channels, skip_channels, max(out_channels // 2, 8))
        self.block = ResidualSEBlock(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(
            x,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        skip = self.attn(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        features: tuple[int, int, int, int] = (64, 128, 256, 512),
        *,
        improved: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.improved = improved

        if not improved:
            f1, f2, f3, f4 = features
            self.stem = LegacyDoubleConv(in_channels, f1)
            self.down1 = LegacyDownBlock(f1, f2)
            self.down2 = LegacyDownBlock(f2, f3)
            self.down3 = LegacyDownBlock(f3, f4)
            self.down4 = LegacyDownBlock(f4, f4 * 2)

            self.up1 = LegacyUpBlock(f4 * 2, f4, f4)
            self.up2 = LegacyUpBlock(f4, f3, f3)
            self.up3 = LegacyUpBlock(f3, f2, f2)
            self.up4 = LegacyUpBlock(f2, f1, f1)
            self.head = nn.Conv2d(f1, num_classes, kernel_size=1)
            return

        f1, f2, f3, f4 = features
        bottleneck_channels = f4 * 2

        self.stem = ResidualSEBlock(in_channels, f1, dropout=0.0)
        self.down1 = EncoderBlock(f1, f2, dropout=0.0)
        self.down2 = EncoderBlock(f2, f3, dropout=dropout * 0.5)
        self.down3 = EncoderBlock(f3, f4, dropout=dropout)
        self.down4 = EncoderBlock(f4, bottleneck_channels, dropout=dropout)
        self.bottleneck = ResidualSEBlock(
            bottleneck_channels,
            bottleneck_channels,
            dropout=dropout,
        )

        self.up1 = DecoderBlock(bottleneck_channels, f4, f4, dropout=dropout)
        self.up2 = DecoderBlock(f4, f3, f3, dropout=dropout * 0.5)
        self.up3 = DecoderBlock(f3, f2, f2, dropout=0.0)
        self.up4 = DecoderBlock(f2, f1, f1, dropout=0.0)
        self.head = nn.Sequential(
            ConvNormAct(f1, f1, kernel_size=3),
            nn.Conv2d(f1, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.improved:
            x5 = self.bottleneck(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.head(x)
