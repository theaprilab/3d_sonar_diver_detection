"""Metric primitives shared by the paper evaluator and unit tests."""

from __future__ import annotations

import torch


def geodesic_rotation_error_degrees(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the SO(3) geodesic error in degrees for paired rotation matrices.

    Both inputs must have shape ``(..., 3, 3)``.  The trace is clamped before
    arccos to make the metric stable around numerically identical rotations.
    """
    relative = prediction.transpose(-1, -2) @ target
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cosine))
