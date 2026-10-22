from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def _up(x: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class UNetPlusPlus(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int] = (64, 128, 256, 512),
        decoder_channels: Sequence[int] = (64, 128, 256, 512),
        num_classes: int = 11,
        deep_supervision: bool = False,
    ):
        super().__init__()
        assert len(in_channels) == 4 and len(decoder_channels) == 4
        c0, c1, c2, c3 = decoder_channels
        self.deep_supervision = deep_supervision

        # Project raw pyramid features to decoder widths (identity if already equal).
        self.in_proj = nn.ModuleList([
            nn.Conv2d(in_channels[i], decoder_channels[i], kernel_size=1) for i in range(4)
        ])

        # X_i_j naming follows the paper: i = depth level (0=highest res), j = column.
        self.conv0_1 = ConvBlock(c0 + c1, c0)
        self.conv1_1 = ConvBlock(c1 + c2, c1)
        self.conv2_1 = ConvBlock(c2 + c3, c2)

        self.conv0_2 = ConvBlock(c0 * 2 + c1, c0)
        self.conv1_2 = ConvBlock(c1 * 2 + c2, c1)

        self.conv0_3 = ConvBlock(c0 * 3 + c1, c0)

        if deep_supervision:
            self.final_1 = nn.Conv2d(c0, num_classes, kernel_size=1)
            self.final_2 = nn.Conv2d(c0, num_classes, kernel_size=1)
            self.final_3 = nn.Conv2d(c0, num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(c0, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        assert len(features) == 4
        x0_0, x1_0, x2_0, x3_0 = [self.in_proj[i](features[i]) for i in range(4)]

        x0_1 = self.conv0_1(torch.cat([x0_0, _up(x1_0, x0_0.shape[-2:])], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, _up(x2_0, x1_0.shape[-2:])], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, _up(x3_0, x2_0.shape[-2:])], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, _up(x1_1, x0_0.shape[-2:])], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, _up(x2_1, x1_0.shape[-2:])], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, _up(x1_2, x0_0.shape[-2:])], dim=1))

        if self.deep_supervision:
            out1 = self.final_1(x0_1)
            out2 = self.final_2(x0_2)
            out3 = self.final_3(x0_3)
            return (out1 + out2 + out3) / 3.0
        return self.final(x0_3)
