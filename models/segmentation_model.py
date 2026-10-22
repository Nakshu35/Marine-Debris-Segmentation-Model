from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import SimpleFeaturePyramidNeck, ViTEncoder
from models.unetpp import UNetPlusPlus


class SegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: ViTEncoder,
        num_classes: int = 11,
        pyramid_channels: Sequence[int] = (64, 128, 256, 512),
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.neck = SimpleFeaturePyramidNeck(encoder.embed_dim, tuple(pyramid_channels))
        self.decoder = UNetPlusPlus(
            in_channels=tuple(pyramid_channels),
            decoder_channels=tuple(pyramid_channels),
            num_classes=num_classes,
            deep_supervision=deep_supervision,
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h_in, w_in = x.shape
        tokens, (h, w) = self.encoder.forward_full(x)
        features = self.neck(tokens, (h, w))
        logits = self.decoder(features)  # stride-4 resolution
        logits = F.interpolate(logits, size=(h_in, w_in), mode="bilinear", align_corners=False)
        return logits

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True
