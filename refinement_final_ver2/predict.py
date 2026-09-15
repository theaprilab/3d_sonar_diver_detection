"""Forward pass over a pre-built ROI batch for the learned measurement models.

``PointGeometryRefiner`` (center and rotation) and
``PointConditionedRotationCovariance`` share the same ``(points, mask,
box_features) -> per-box array`` calling convention, so a single wrapper
covers both.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def predict_frame(model: torch.nn.Module, frame: dict, device: str) -> np.ndarray:
    """Run ``model`` over every ROI in ``frame`` (as built by ``learned_data``
    / ``learned_center_data_v2``) and return the per-box output as numpy.
    """
    if not frame["detections"]:
        # PointGeometryRefiner outputs 9 values; covariance models output 3.
        out_width = 9 if hasattr(model, "center_head") else 3
        return np.empty((0, out_width), dtype=np.float32)
    points = torch.from_numpy(frame["point_features"]).to(device)
    masks = torch.from_numpy(frame["point_mask"]).to(device)
    metadata = torch.from_numpy(frame["box_features"]).to(device)
    return model(points, masks, metadata).cpu().numpy()
