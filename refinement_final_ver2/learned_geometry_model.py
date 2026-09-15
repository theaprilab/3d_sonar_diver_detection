"""Small current-frame point-conditioned residual model for Stage G."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from learned_data import BOX_FEATURES, POINT_FEATURES, TARGET_DIM


@dataclass(frozen=True)
class GeometryModelConfig:
    point_hidden: int = 64
    point_embedding: int = 128
    fusion_hidden: int = 128
    dropout: float = 0.10
    center_bound_m: tuple[float, float, float] = (0.50, 0.50, 0.50)
    log_dimension_bound: tuple[float, float, float] = (0.70, 0.70, 0.70)
    rotation_bound_rad: tuple[float, float, float] = (
        1.0471975512, 1.0471975512, 1.0471975512)

    def payload(self) -> dict:
        return asdict(self)


STAGE_G_MODEL_CONFIG = GeometryModelConfig()


class PointGeometryRefiner(nn.Module):
    def __init__(self, config: GeometryModelConfig = STAGE_G_MODEL_CONFIG):
        super().__init__()
        self.config = config
        self.point_mlp = nn.Sequential(
            nn.Linear(POINT_FEATURES, config.point_hidden), nn.ReLU(inplace=True),
            nn.Linear(config.point_hidden, config.point_embedding), nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * config.point_embedding + BOX_FEATURES, config.fusion_hidden),
            nn.ReLU(inplace=True), nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden, config.fusion_hidden // 2), nn.ReLU(inplace=True),
        )
        width = config.fusion_hidden // 2
        self.center_head = nn.Linear(width, 3)
        self.dimension_head = nn.Linear(width, 3)
        self.rotation_head = nn.Linear(width, 3)
        self.register_buffer("output_bounds", torch.tensor(
            config.center_bound_m + config.log_dimension_bound + config.rotation_bound_rad,
            dtype=torch.float32))
        self._initialize_identity()

    def _initialize_identity(self) -> None:
        # A fresh add-on is exactly the frozen detector, not a random corruption.
        for head in (self.center_head, self.dimension_head, self.rotation_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, points: torch.Tensor, mask: torch.Tensor,
                box_features: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or mask.shape != points.shape[:2]:
            raise ValueError("expected points [B,N,F] and mask [B,N]")
        embeddings = self.point_mlp(points)
        valid = mask.unsqueeze(-1).bool()
        negative = torch.finfo(embeddings.dtype).min
        maximum = embeddings.masked_fill(~valid, negative).amax(dim=1)
        has_points = valid.any(dim=1)
        maximum = torch.where(has_points, maximum, torch.zeros_like(maximum))
        mean = (embeddings * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        fused = self.fusion(torch.cat([maximum, mean, box_features], dim=-1))
        raw = torch.cat([
            self.center_head(fused), self.dimension_head(fused), self.rotation_head(fused)
        ], dim=-1)
        if raw.shape[-1] != TARGET_DIM:
            raise AssertionError("Stage-G output width mismatch")
        return torch.tanh(raw) * self.output_bounds


def normalized_smooth_l1(prediction: torch.Tensor, target: torch.Tensor,
                         bounds: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scaled_prediction = prediction / bounds
    scaled_target = target / bounds
    losses = {
        "center": F.smooth_l1_loss(scaled_prediction[:, :3], scaled_target[:, :3]),
        "dimensions": F.smooth_l1_loss(scaled_prediction[:, 3:6], scaled_target[:, 3:6]),
        "rotation": F.smooth_l1_loss(scaled_prediction[:, 6:9], scaled_target[:, 6:9]),
    }
    return sum(losses.values()), losses
