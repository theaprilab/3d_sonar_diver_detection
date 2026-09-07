"""Target conversions shared by head/backbone ablation recipes."""

from __future__ import annotations

import torch

from sonarvoxnet.models.rotation import matrix_to_sixd


def project_to_zyaw(rotation: torch.Tensor) -> torch.Tensor:
    """Project an SO(3) matrix to its BEV yaw-only rotation matrix.

    This makes the center and anchor z-yaw ablations comparable: both receive
    a 6D representation of the same yaw-only target, not the full 3D pose.
    """
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must end in a 3-by-3 matrix.")
    yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    result = torch.zeros_like(rotation)
    result[..., 0, 0] = cosine
    result[..., 0, 1] = -sine
    result[..., 1, 0] = sine
    result[..., 1, 1] = cosine
    result[..., 2, 2] = 1
    return result


def rotation_target(rotation: torch.Tensor, mode: str) -> torch.Tensor:
    """Create the documented target representation for a head recipe."""
    if mode == "full_3d":
        return matrix_to_sixd(rotation)
    if mode == "zyaw":
        return matrix_to_sixd(project_to_zyaw(rotation))
    raise ValueError(f"Unsupported rotation target mode: {mode}")
