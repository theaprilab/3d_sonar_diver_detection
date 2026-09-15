"""Point-conditioned Lie-tangent uncertainty for frozen rotation correction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
import torch.nn as nn

from bayesian_box_filter import rotation_innovation, so3_exp
from learned_data import BOX_FEATURES, POINT_FEATURES


@dataclass(frozen=True)
class RotationUncertaintyConfig:
    point_hidden: int = 32
    point_embedding: int = 64
    fusion_hidden: int = 64
    variance_floor_rad2: float = math.radians(0.5) ** 2
    log_variance_delta_bound: float = 4.0

    def payload(self) -> dict:
        return asdict(self)


ROTATION_UNCERTAINTY_CONFIG = RotationUncertaintyConfig()


class PointConditionedRotationCovariance(nn.Module):
    """Predict diagonal covariance of a frozen mean's local SO(3) error."""

    def __init__(self, global_variance,
                 config: RotationUncertaintyConfig = ROTATION_UNCERTAINTY_CONFIG):
        super().__init__()
        self.config = config
        variance = torch.as_tensor(global_variance, dtype=torch.float32).clamp_min(
            config.variance_floor_rad2)
        if variance.shape != (3,):
            raise ValueError("global rotation variance must have shape [3]")
        self.register_buffer("global_log_variance", torch.log(variance))
        self.point_mlp = nn.Sequential(
            nn.Linear(POINT_FEATURES, config.point_hidden), nn.ReLU(inplace=True),
            nn.Linear(config.point_hidden, config.point_embedding), nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * config.point_embedding + BOX_FEATURES, config.fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.fusion_hidden, config.fusion_hidden), nn.ReLU(inplace=True),
        )
        self.log_variance_delta = nn.Linear(config.fusion_hidden, 3)
        nn.init.zeros_(self.log_variance_delta.weight)
        nn.init.zeros_(self.log_variance_delta.bias)

    def forward(self, points, mask, box_features):
        embeddings = self.point_mlp(points)
        valid = mask.unsqueeze(-1).bool()
        negative = torch.finfo(embeddings.dtype).min
        maximum = embeddings.masked_fill(~valid, negative).amax(dim=1)
        maximum = torch.where(valid.any(dim=1), maximum, torch.zeros_like(maximum))
        mean = (embeddings * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        fused = self.fusion(torch.cat([maximum, mean, box_features], dim=-1))
        delta = torch.tanh(self.log_variance_delta(fused)) * \
            self.config.log_variance_delta_bound
        return torch.exp(self.global_log_variance + delta).clamp_min(
            self.config.variance_floor_rad2)


def gaussian_diagonal_nll(error, variance):
    return 0.5 * torch.mean(torch.sum(
        error.square() / variance + torch.log(variance)
        + math.log(2.0 * math.pi), dim=-1))


def lie_correction_error(mean, target) -> np.ndarray:
    """Right-invariant, z-pi-symmetry-aware error of tangent corrections."""
    mean = np.asarray(mean, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mean.shape != target.shape or mean.ndim != 2 or mean.shape[1] != 3:
        raise ValueError("mean and target must have equal [N,3] shape")
    return np.asarray([
        rotation_innovation(so3_exp(prediction), so3_exp(label), fold_z_pi=True)
        for prediction, label in zip(mean, target)
    ], dtype=np.float32)


def bayesian_lie_correction(mean, prior_variance, error_variance):
    """First-order zero-prior Gaussian shrinkage in detector-local tangent."""
    mean = np.asarray(mean, dtype=np.float64)
    prior = np.asarray(prior_variance, dtype=np.float64)
    error = np.asarray(error_variance, dtype=np.float64)
    if mean.ndim != 2 or mean.shape[1] != 3 or prior.shape != (3,):
        raise ValueError("expected mean [N,3] and prior variance [3]")
    error = np.broadcast_to(error, mean.shape)
    if np.any(prior <= 0.0) or np.any(error <= 0.0):
        raise ValueError("rotation variances must be positive")
    gain = prior / (prior + error)
    posterior_variance = prior * error / (prior + error)
    return ((gain * mean).astype(np.float32), gain.astype(np.float32),
            posterior_variance.astype(np.float32))
