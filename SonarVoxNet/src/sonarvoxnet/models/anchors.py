"""Anchor-grid and box-regression utilities for the anchor-head ablation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AnchorGridSpec:
    """Geometry shared by every run using the anchor-head ablation.

    The grid is expressed in the detector BEV frame.  ``sizes`` contains one
    ``(length, width, height)`` tuple per anchor template and ``yaws`` gives
    the corresponding in-plane template orientations.
    """

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_center: float
    stride: tuple[float, float]
    sizes: tuple[tuple[float, float, float], ...]
    yaws: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.sizes) != len(self.yaws):
            raise ValueError("Each anchor size must have one yaw template.")
        if not self.sizes:
            raise ValueError("At least one anchor template is required.")


DEFAULT_ANCHOR_GRID = AnchorGridSpec(
    x_range=(0.0, 12.0),
    y_range=(-5.0, 5.0),
    z_center=0.0,
    stride=(0.2, 0.2),
    sizes=((1.6, 0.7, 1.7), (1.6, 0.7, 1.7)),
    yaws=(0.0, 1.5707963267948966),
)


def build_anchor_grid(
    height: int,
    width: int,
    spec: AnchorGridSpec = DEFAULT_ANCHOR_GRID,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build anchors as ``(height, width, templates, 7)`` tensors.

    The final dimension is ``(x, y, z, length, width, height, yaw)``.  The
    feature-map shape is an explicit argument so decoding remains valid when a
    backbone changes its BEV resolution.
    """
    if height < 1 or width < 1:
        raise ValueError("Anchor-grid height and width must be positive.")
    x = spec.x_range[0] + (torch.arange(width, device=device, dtype=dtype) + 0.5) * spec.stride[0]
    y = spec.y_range[0] + (torch.arange(height, device=device, dtype=dtype) + 0.5) * spec.stride[1]
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
    templates = torch.tensor(
        [(*size, yaw) for size, yaw in zip(spec.sizes, spec.yaws)], device=device, dtype=dtype
    )
    anchors = torch.empty(height, width, len(spec.sizes), 7, device=device, dtype=dtype)
    anchors[..., 0] = x_grid.unsqueeze(-1)
    anchors[..., 1] = y_grid.unsqueeze(-1)
    anchors[..., 2] = spec.z_center
    anchors[..., 3:] = templates
    return anchors


def encode_box_residuals(anchors: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Encode matched boxes relative to anchors.

    Inputs use ``(x, y, z, length, width, height, ...)``.  Any trailing
    rotation target is preserved unchanged, allowing the ablation to regress a
    6D target while anchor placement remains z-yaw based.
    """
    if anchors.shape[-1] != 7 or boxes.shape[-1] < 7:
        raise ValueError("Anchors need 7 values and boxes need at least 7 values.")
    diagonal = torch.linalg.vector_norm(anchors[..., 3:5], dim=-1).clamp_min(1e-6)
    residuals = torch.empty_like(boxes)
    residuals[..., 0] = (boxes[..., 0] - anchors[..., 0]) / diagonal
    residuals[..., 1] = (boxes[..., 1] - anchors[..., 1]) / diagonal
    residuals[..., 2] = (boxes[..., 2] - anchors[..., 2]) / anchors[..., 5].clamp_min(1e-6)
    residuals[..., 3:6] = torch.log(boxes[..., 3:6] / anchors[..., 3:6].clamp_min(1e-6))
    residuals[..., 6:] = boxes[..., 6:]
    return residuals


def decode_box_residuals(anchors: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
    """Decode anchor-relative residuals back to metric box parameters."""
    if anchors.shape[-1] != 7 or residuals.shape[-1] < 6:
        raise ValueError("Anchors need 7 values and residuals need at least 6 values.")
    diagonal = torch.linalg.vector_norm(anchors[..., 3:5], dim=-1).clamp_min(1e-6)
    boxes = torch.empty_like(residuals)
    boxes[..., 0] = residuals[..., 0] * diagonal + anchors[..., 0]
    boxes[..., 1] = residuals[..., 1] * diagonal + anchors[..., 1]
    boxes[..., 2] = residuals[..., 2] * anchors[..., 5] + anchors[..., 2]
    boxes[..., 3:6] = torch.exp(residuals[..., 3:6]).clamp_max(1e6) * anchors[..., 3:6]
    boxes[..., 6:] = residuals[..., 6:]
    return boxes
