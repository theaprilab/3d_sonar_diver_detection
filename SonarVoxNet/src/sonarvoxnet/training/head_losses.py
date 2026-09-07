"""Losses whose choice is fixed within the head/backbone ablation."""

from __future__ import annotations

import torch
from torch.nn import functional as functional


def anchor_head_loss(
    prediction: dict[str, torch.Tensor],
    objectness_target: torch.Tensor,
    regression_target: torch.Tensor,
    positive_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute focal objectness and Smooth-L1 regression on positive anchors."""
    logits = prediction["objectness"]
    probability = torch.sigmoid(logits)
    target = objectness_target.to(dtype=logits.dtype)
    alpha = torch.where(target > 0, 0.25, 0.75)
    focal_weight = alpha * (target - probability).abs().pow(2)
    classification = (focal_weight * functional.binary_cross_entropy_with_logits(logits, target, reduction="none")).sum()
    classification = classification / positive_mask.sum().clamp_min(1)

    residual = prediction["regression"]
    target_residual = regression_target.to(dtype=residual.dtype)
    regression = functional.smooth_l1_loss(residual, target_residual, reduction="none")
    regression = (regression * positive_mask[:, :, None, :, :]).sum() / positive_mask.sum().clamp_min(1)
    return {"classification": classification, "regression": regression, "total": classification + regression}
