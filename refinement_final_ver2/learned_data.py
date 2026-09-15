"""Stage-G object-centric ROI and supervision construction.

The frozen detector supplies boxes.  Ground truth is used only to create
one-to-one residual labels on the refiner fit split.  Inference needs only a
box and the current frame's retained sonar points.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

import _paths  # noqa: F401
import eval_voxelnet as E
from bayesian_box_filter import rotation_innovation


POINT_FEATURES = 7
BOX_FEATURES = 10
TARGET_DIM = 9


@dataclass(frozen=True)
class RoiConfig:
    max_points: int = 256
    context_scale: float = 1.5
    minimum_half_extent_m: float = 0.30
    center_match_gate_m: float = 1.0
    matching_iou_samples: int = 1024
    center_bound_m: tuple[float, float, float] = (0.50, 0.50, 0.50)
    log_dimension_bound: tuple[float, float, float] = (0.70, 0.70, 0.70)
    rotation_bound_rad: tuple[float, float, float] = (
        np.pi / 3.0, np.pi / 3.0, np.pi / 3.0)

    def payload(self) -> dict:
        return asdict(self)


STAGE_G_ROI_CONFIG = RoiConfig()


def retained_points(voxel_features: np.ndarray, num_points: np.ndarray) -> np.ndarray:
    """Recover unique retained raw [x,y,z,intensity] points from voxel cache."""
    if len(voxel_features) == 0:
        return np.empty((0, 4), dtype=np.float32)
    width = voxel_features.shape[-1]
    if width < 4:
        raise ValueError(f"expected xyz+intensity voxel features, got width={width}")
    valid = [voxel_features[index, : int(count), :4]
             for index, count in enumerate(num_points) if int(count) > 0]
    return (np.concatenate(valid, axis=0).astype(np.float32, copy=False)
            if valid else np.empty((0, 4), dtype=np.float32))


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def object_roi(box: dict, points_xyzi: np.ndarray,
               config: RoiConfig = STAGE_G_ROI_CONFIG) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return padded point features, mask and detector-only metadata."""
    center, dims, rotation = E.pred_obb_from_box(box)
    center = np.asarray(center, dtype=np.float64)
    dims = np.maximum(np.asarray(dims, dtype=np.float64), 1e-3)
    rotation = np.asarray(rotation, dtype=np.float64)
    xyz = np.asarray(points_xyzi[:, :3], dtype=np.float64)
    local = (xyz - center) @ rotation if len(xyz) else np.empty((0, 3))
    half = np.maximum(0.5 * config.context_scale * dims,
                      config.minimum_half_extent_m)
    keep = np.all(np.abs(local) <= half, axis=1) if len(local) else np.zeros(0, bool)
    local, raw = local[keep], points_xyzi[keep]

    # Deterministic, order-preserving coverage sampling.  No inference RNG.
    if len(local) > config.max_points:
        indices = np.linspace(0, len(local) - 1, config.max_points).round().astype(int)
        local, raw = local[indices], raw[indices]
    count = len(local)
    point_features = np.zeros((config.max_points, POINT_FEATURES), dtype=np.float32)
    mask = np.zeros(config.max_points, dtype=np.float32)
    if count:
        normalized = local / np.maximum(0.5 * dims, 1e-3)
        intensity = _signed_log1p(raw[:, 3:4].astype(np.float64))
        point_features[:count] = np.concatenate([local, normalized, intensity], axis=1)
        mask[:count] = 1.0
        coverage = np.minimum(np.ptp(local, axis=0) / dims, 2.0)
    else:
        coverage = np.zeros(3, dtype=np.float64)
    metadata = np.array([
        *np.log(dims),
        np.hypot(center[0], center[1]) / 12.0,
        center[2] / 2.5,
        float(box.get("score", 0.0)),
        np.log1p(count) / np.log1p(config.max_points),
        *coverage,
    ], dtype=np.float32)
    if metadata.shape != (BOX_FEATURES,):
        raise AssertionError(f"metadata shape mismatch: {metadata.shape}")
    return point_features, mask, metadata


def match_detections(boxes: list[dict], gt_boxes: np.ndarray,
                     config: RoiConfig = STAGE_G_ROI_CONFIG) -> dict[int, int]:
    """Deterministic one-to-one assignment, IoU cost inside a fixed center gate."""
    if not boxes or len(gt_boxes) == 0:
        return {}
    gt_obbs = [E.gt_obb_from_row(row) for row in np.asarray(gt_boxes)]
    cost = np.full((len(boxes), len(gt_obbs)), 1e6, dtype=np.float64)
    rng = np.random.default_rng(0)
    for box_index, box in enumerate(boxes):
        center, dims, rotation = E.pred_obb_from_box(box)
        for gt_index, (gt_center, gt_dims, gt_rotation) in enumerate(gt_obbs):
            distance = float(np.linalg.norm(np.asarray(center) - np.asarray(gt_center)))
            if distance > config.center_match_gate_m:
                continue
            iou = E.iou_3d_obb(
                center, dims, rotation, gt_center, gt_dims, gt_rotation,
                n_samples=config.matching_iou_samples, rng=rng)
            # Tiny distance term resolves equal/zero Monte-Carlo IoUs deterministically.
            cost[box_index, gt_index] = 1.0 - iou + 1e-4 * distance
    rows, columns = linear_sum_assignment(cost)
    return {int(row): int(column) for row, column in zip(rows, columns)
            if cost[row, column] < 1e5}


def residual_target(box: dict, gt_row: np.ndarray) -> np.ndarray:
    """Local-center, log-dimension and symmetry-aware right rotation residual."""
    center, dims, rotation = E.pred_obb_from_box(box)
    gt_center, gt_dims, gt_rotation = E.gt_obb_from_row(np.asarray(gt_row))
    center_local = np.asarray(rotation).T @ (np.asarray(gt_center) - np.asarray(center))
    log_dimensions = np.log(np.maximum(gt_dims, 1e-6)) - np.log(np.maximum(dims, 1e-6))
    rotation_local = rotation_innovation(rotation, gt_rotation, fold_z_pi=True)
    target = np.concatenate([center_local, log_dimensions, rotation_local]).astype(np.float32)
    if target.shape != (TARGET_DIM,) or not np.all(np.isfinite(target)):
        raise ValueError("invalid Stage-G residual target")
    return target


def build_frame_record(frame_id: int, boxes: Iterable[dict], gt_boxes: np.ndarray,
                       points_xyzi: np.ndarray,
                       config: RoiConfig = STAGE_G_ROI_CONFIG) -> dict:
    """Construct all inference ROIs and matched training labels for one frame."""
    boxes = [dict(box) for box in boxes]
    matches = match_detections(boxes, gt_boxes, config)
    rois, masks, metadata, targets, matched, matched_gt_indices = [], [], [], [], [], []
    for index, box in enumerate(boxes):
        roi, mask, meta = object_roi(box, points_xyzi, config)
        rois.append(roi); masks.append(mask); metadata.append(meta)
        if index in matches:
            targets.append(residual_target(box, gt_boxes[matches[index]]))
            matched.append(True)
            matched_gt_indices.append(matches[index])
        else:
            targets.append(np.zeros(TARGET_DIM, dtype=np.float32))
            matched.append(False)
            matched_gt_indices.append(-1)
    n = len(boxes)
    return {
        "frame_id": int(frame_id),
        "detections": boxes,
        "gt_boxes": np.asarray(gt_boxes),
        "point_features": np.stack(rois) if n else np.empty(
            (0, config.max_points, POINT_FEATURES), np.float32),
        "point_mask": np.stack(masks) if n else np.empty((0, config.max_points), np.float32),
        "box_features": np.stack(metadata) if n else np.empty((0, BOX_FEATURES), np.float32),
        "targets": np.stack(targets) if n else np.empty((0, TARGET_DIM), np.float32),
        "matched": np.asarray(matched, dtype=bool),
        "matched_gt_index": np.asarray(matched_gt_indices, dtype=np.int64),
    }


def validate_scene_blob(blob: dict, expected_scenes: int | None = None) -> None:
    if blob.get("split") != "val_full":
        raise ValueError(f"Stage-G LOSO requires val_full, got {blob.get('split')}")
    scenes = blob.get("scenes", {})
    if expected_scenes is not None and len(scenes) != expected_scenes:
        raise ValueError(f"expected {expected_scenes} scenes, got {len(scenes)}")
    for scene, frames in scenes.items():
        previous = None
        for frame in frames:
            if previous is not None and frame["frame_id"] <= previous:
                raise ValueError(f"non-increasing frames in {scene}")
            previous = frame["frame_id"]
            n = len(frame["detections"])
            if frame["point_features"].shape[0] != n or frame["targets"].shape != (n, TARGET_DIM):
                raise ValueError(f"record shape mismatch in {scene}/{previous}")


def matched_arrays(blob: dict, fit_scenes: Iterable[str]) -> tuple[np.ndarray, ...]:
    """Flatten matched samples from exactly the supplied scene names."""
    point_features, masks, box_features, targets = [], [], [], []
    for scene in fit_scenes:
        for frame in blob["scenes"][scene]:
            selected = frame["matched"]
            if np.any(selected):
                point_features.append(frame["point_features"][selected])
                masks.append(frame["point_mask"][selected])
                box_features.append(frame["box_features"][selected])
                targets.append(frame["targets"][selected])
    if not targets:
        raise ValueError("no matched Stage-G samples")
    return tuple(np.concatenate(values, axis=0)
                 for values in (point_features, masks, box_features, targets))
