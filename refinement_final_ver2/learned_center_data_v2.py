"""Cap-safe object-centric ROI schema for the one-shot center specialist.

V1 checkpoints continue to use learned_data.object_roi. This module versions
the changed metadata semantics and spatial sampling separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

import _paths  # noqa: F401
import eval_voxelnet as E
from learned_data import (BOX_FEATURES, POINT_FEATURES, STAGE_G_ROI_CONFIG,
                          match_detections, residual_target)


@dataclass(frozen=True)
class CenterRoiV2Config:
    max_points: int = 256
    context_scale: float = 1.5
    minimum_half_extent_m: float = 0.30
    count_normalizer: int = 8192
    spatial_bins_per_axis: int = 8
    schema_version: str = "center-roi-cap-safe-v2"

    def payload(self) -> dict:
        return asdict(self)


CENTER_ROI_V2_CONFIG = CenterRoiV2Config()


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def spatial_coverage_indices(local: np.ndarray, intensity: np.ndarray,
                             half: np.ndarray, max_points: int,
                             bins_per_axis: int) -> np.ndarray:
    """Deterministically retain broad spatial coverage before filling density."""
    n = len(local)
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    normalized = np.clip((local / np.maximum(half, 1e-6) + 1.0) * 0.5, 0.0, 1.0)
    quantized = np.minimum((normalized * bins_per_axis).astype(np.int64),
                           bins_per_axis - 1)
    code = (quantized[:, 0] + bins_per_axis * quantized[:, 1]
            + bins_per_axis**2 * quantized[:, 2])
    order = np.lexsort((np.arange(n), -np.abs(intensity), code))
    ordered_code = code[order]
    first = np.r_[True, ordered_code[1:] != ordered_code[:-1]]
    representatives = order[first]
    if len(representatives) >= max_points:
        positions = np.linspace(0, len(representatives) - 1, max_points).round().astype(int)
        return representatives[positions]
    chosen = np.zeros(n, dtype=bool)
    chosen[representatives] = True
    remaining = order[~chosen[order]]
    needed = max_points - len(representatives)
    positions = np.linspace(0, len(remaining) - 1, needed).round().astype(int)
    return np.concatenate([representatives, remaining[positions]])


def object_roi_center_v2(box: dict, points_xyzi: np.ndarray,
                         config: CenterRoiV2Config = CENTER_ROI_V2_CONFIG):
    center, dims, rotation = E.pred_obb_from_box(box)
    center = np.asarray(center, dtype=np.float64)
    dims = np.maximum(np.asarray(dims, dtype=np.float64), 1e-3)
    rotation = np.asarray(rotation, dtype=np.float64)
    xyz = np.asarray(points_xyzi[:, :3], dtype=np.float64)
    local_all = ((xyz - center) @ rotation if len(xyz)
                 else np.empty((0, 3), dtype=np.float64))
    half = np.maximum(0.5 * config.context_scale * dims,
                      config.minimum_half_extent_m)
    keep = (np.all(np.abs(local_all) <= half, axis=1)
            if len(local_all) else np.zeros(0, dtype=bool))
    local_full = local_all[keep]
    raw_full = points_xyzi[keep]
    raw_count = len(local_full)
    coverage = (np.minimum(np.ptp(local_full, axis=0) / dims, 2.0)
                if raw_count else np.zeros(3, dtype=np.float64))
    if raw_count > config.max_points:
        indices = spatial_coverage_indices(
            local_full, raw_full[:, 3], half, config.max_points,
            config.spatial_bins_per_axis)
        local, raw = local_full[indices], raw_full[indices]
    else:
        local, raw = local_full, raw_full
    sampled_count = len(local)
    features = np.zeros((config.max_points, POINT_FEATURES), dtype=np.float32)
    mask = np.zeros(config.max_points, dtype=np.float32)
    if sampled_count:
        normalized = local / np.maximum(0.5 * dims, 1e-3)
        intensity = _signed_log1p(raw[:, 3:4].astype(np.float64))
        features[:sampled_count] = np.concatenate([local, normalized, intensity], axis=1)
        mask[:sampled_count] = 1.0
    metadata = np.array([
        *np.log(dims), np.hypot(center[0], center[1]) / 12.0, center[2] / 2.5,
        float(box.get("score", 0.0)),
        np.clip(np.log1p(raw_count) / np.log1p(config.count_normalizer), 0.0, 1.0),
        *coverage,
    ], dtype=np.float32)
    if metadata.shape != (BOX_FEATURES,):
        raise AssertionError("center V2 metadata width changed")
    return features, mask, metadata, raw_count


def build_center_frame_v2(frame_id: int, boxes: list[dict], gt_boxes: np.ndarray,
                          points_xyzi: np.ndarray,
                          config: CenterRoiV2Config = CENTER_ROI_V2_CONFIG) -> dict:
    boxes = [dict(box) for box in boxes]
    matches = match_detections(boxes, gt_boxes, STAGE_G_ROI_CONFIG)
    rois, masks, metadata, raw_counts = [], [], [], []
    targets, matched, matched_gt_indices = [], [], []
    for index, box in enumerate(boxes):
        roi, mask, meta, raw_count = object_roi_center_v2(box, points_xyzi, config)
        rois.append(roi); masks.append(mask); metadata.append(meta); raw_counts.append(raw_count)
        if index in matches:
            targets.append(residual_target(box, gt_boxes[matches[index]]))
            matched.append(True); matched_gt_indices.append(matches[index])
        else:
            targets.append(np.zeros(9, dtype=np.float32))
            matched.append(False); matched_gt_indices.append(-1)
    n = len(boxes)
    return {
        "frame_id": int(frame_id), "detections": boxes, "gt_boxes": np.asarray(gt_boxes),
        "point_features": np.stack(rois) if n else np.empty(
            (0, config.max_points, POINT_FEATURES), np.float32),
        "point_mask": np.stack(masks) if n else np.empty((0, config.max_points), np.float32),
        "box_features": np.stack(metadata) if n else np.empty((0, BOX_FEATURES), np.float32),
        "raw_roi_count": np.asarray(raw_counts, dtype=np.int32),
        "targets": np.stack(targets) if n else np.empty((0, 9), np.float32),
        "matched": np.asarray(matched, dtype=bool),
        "matched_gt_index": np.asarray(matched_gt_indices, dtype=np.int64),
    }
