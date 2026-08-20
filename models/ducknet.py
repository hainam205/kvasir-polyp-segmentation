from __future__ import annotations
import torch
import torch.nn as nn


# ===== Basic Convolution Block =====
class ConvBNActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


# ===== Residual Block =====
class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBNActivation(channels, channels)
        self.conv2 = ConvBNActivation(channels, channels)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


# ===== Multi-Branch Feature Extraction (Duck-inspired) =====
class MultiBranchFeatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        branch_channels = max(out_channels // 4, 8)

        self.input_projection = ConvBNActivation(
            in_channels, branch_channels, kernel_size=1
        )

        # Branch 1: Residual
        self.branch_residual = ResidualConvBlock(branch_channels)

        # Branch 2: Medium receptive field
        self.branch_medium = nn.Sequential(
            ConvBNActivation(branch_channels, branch_channels, dilation=1),
            ConvBNActivation(branch_channels, branch_channels, dilation=2),
        )

        # Branch 3: Wide receptive field
        self.branch_wide = nn.Sequential(
            ConvBNActivation(branch_channels, branch_channels, dilation=2),
            ConvBNActivation(branch_channels, branch_channels, dilation=3),
        )

        # Branch 4: Separable convolution
        self.branch_separable = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, (1, 5),
                      padding=(0, 2), bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
            nn.Conv2d(branch_channels, branch_channels, (5, 1),
                      padding=(2, 0), bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
        )

        # Fusion
        self.fusion = nn.Sequential(
            ConvBNActivation(branch_channels * 4, out_channels, kernel_size=1),
            ConvBNActivation(out_channels, out_channels, kernel_size=3),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.shortcut = ConvBNActivation(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        projected = self.input_projection(x)

        b1 = self.branch_residual(projected)
        b2 = self.branch_medium(projected)
        b3 = self.branch_wide(projected)
        b4 = self.branch_separable(projected)

        merged = torch.cat([b1, b2, b3, b4], dim=1)
        fused = self.fusion(merged)
        fused = self.dropout(fused)

        return fused + self.shortcut(x)


# ===== Downsampling Block =====
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.down = ConvBNActivation(in_channels, out_channels, stride=2)
        self.feature_block = MultiBranchFeatureBlock(out_channels, out_channels, dropout=dropout)

    def forward(self, x):
        x = self.down(x)
        return self.feature_block(x)


# ===== Upsampling Block =====
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout=0.0):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.channel_adjust = ConvBNActivation(in_channels, out_channels)
        self.merge = ConvBNActivation(out_channels + skip_channels, out_channels, kernel_size=1)

        self.feature_block = MultiBranchFeatureBlock(out_channels, out_channels, dropout=dropout)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = self.channel_adjust(x)

        # Concatenate skip features to preserve fine boundaries.
        x = torch.cat([x, skip], dim=1)
        x = self.merge(x)

        return self.feature_block(x)


# ===== Main Model =====
class DuckNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_channels=32, dropout=0.1):
        super().__init__()

        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2

        # Encoder
        self.initial_block = MultiBranchFeatureBlock(in_channels, c1, dropout=0.0)
        self.encoder_stage1 = DownsampleBlock(c1, c2, dropout=0.0)
        self.encoder_stage2 = DownsampleBlock(c2, c3, dropout=dropout * 0.5)
        self.encoder_stage3 = DownsampleBlock(c3, c4, dropout=dropout)
        self.encoder_stage4 = DownsampleBlock(c4, c5, dropout=dropout)

        # Bottleneck
        self.bottleneck = MultiBranchFeatureBlock(c5, c5, dropout=dropout)

        # Decoder
        self.decoder_stage1 = UpsampleBlock(c5, c4, c4, dropout=dropout)
        self.decoder_stage2 = UpsampleBlock(c4, c3, c3, dropout=dropout * 0.5)
        self.decoder_stage3 = UpsampleBlock(c3, c2, c2, dropout=dropout * 0.25)
        self.decoder_stage4 = UpsampleBlock(c2, c1, c1, dropout=0.0)

        # Output
        self.output_layer = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.initial_block(x)
        x2 = self.encoder_stage1(x1)
        x3 = self.encoder_stage2(x2)
        x4 = self.encoder_stage3(x3)
        x5 = self.encoder_stage4(x4)

        x = self.bottleneck(x5)

        x = self.decoder_stage1(x, x4)
        x = self.decoder_stage2(x, x3)
        x = self.decoder_stage3(x, x2)
        x = self.decoder_stage4(x, x1)

        return self.output_layer(x)