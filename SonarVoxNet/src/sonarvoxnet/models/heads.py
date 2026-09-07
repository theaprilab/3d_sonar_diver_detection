"""Detection heads used by the center-versus-anchor ablation."""

from __future__ import annotations

import math

import torch
from torch import nn

from sonarvoxnet.models.rotation import CHANNELS


class AnchorHead(nn.Module):
    """One-stage anchor head with objectness and metric-box residual outputs.

    Regression channels are ``(dx, dy, dz, dlog_lwh, rotation)``.  The anchor
    templates encode only BEV yaw, while the rotation branch predicts the
    selected full-3D representation for a matched object.
    """

    def __init__(self, channels: int = 128, num_anchors: int = 2, rotation_representation: str = "6d") -> None:
        super().__init__()
        if num_anchors < 1:
            raise ValueError("num_anchors must be positive.")
        if rotation_representation not in CHANNELS:
            raise ValueError(f"Unsupported rotation representation: {rotation_representation}")
        self.num_anchors = num_anchors
        self.rotation_channels = CHANNELS[rotation_representation]
        self.objectness = nn.Conv2d(channels, num_anchors, 1)
        self.regression = nn.Conv2d(channels, num_anchors * (6 + self.rotation_channels), 1)
        nn.init.constant_(self.objectness.bias, -math.log(99.0))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, _, height, width = features.shape
        regression = self.regression(features).reshape(
            batch, self.num_anchors, 6 + self.rotation_channels, height, width
        )
        return {"objectness": self.objectness(features), "regression": regression}
