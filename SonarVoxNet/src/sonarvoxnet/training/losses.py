"""Losses used by the public center-head training path."""

from __future__ import annotations

import torch
from torch.nn import functional as functional


def rotation_target_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, mode: str = "l1_target") -> torch.Tensor:
    """Compute target-space rotation loss only at valid center locations.

    The proposed recipe uses ``l1_target`` with a 6D target.  ``mse_target``
    is retained for the raw 9D rotation-matrix ablation.
    """
    if mode == "l1_target":
        per_value = functional.l1_loss(prediction, target, reduction="none")
    elif mode == "mse_target":
        per_value = functional.mse_loss(prediction, target, reduction="none")
    else:
        raise ValueError(f"Unsupported rotation loss mode: {mode}")
    return (per_value * mask.unsqueeze(-1)).sum() / mask.sum().clamp_min(1)
