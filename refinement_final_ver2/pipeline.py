"""End-to-end reference glue: detector boxes + raw points -> refined boxes.

Mirrors the per-frame algorithm in the confirmatory ``final_eval/modal_final_apply.py``
runner, minus the Modal/dump-replay plumbing. Two learned measurement models
produce the per-frame residuals and covariance that ``FrozenGeometryRefiner``
(the causal numpy filtering core) consumes; a third, training-free routine
produces the size measurement. Fitted weights (``center_allval.pt`` /
``rotation_allval.pt`` / ``covariance_allval.pt`` per seed) are distributed
separately and are not part of this code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from frozen_geometry_runtime import (
    DetectionGeometryMeasurement, ExtentMeasurement,
    FrameGeometryMeasurements, FrozenGeometryRefiner)
from learned_center_data_v2 import build_center_frame_v2
from learned_data import build_frame_record
from learned_geometry_model import GeometryModelConfig, PointGeometryRefiner
from learned_rotation_uncertainty import PointConditionedRotationCovariance
from predict import predict_frame
from training_free_filter import FROZEN_REFINEMENT_BASELINE, point_support_dimensions

_NO_GT = np.empty((0, 9), dtype=np.float32)


@dataclass
class GeometryModels:
    center: PointGeometryRefiner
    rotation: PointGeometryRefiner
    covariance: PointConditionedRotationCovariance
    temperature: float


def load_geometry_models(fit_dir: Path, device: str) -> GeometryModels:
    """Load one seed's fitted weights, as written by ``modal_final_fit.py``."""
    def _load_refiner(name: str) -> PointGeometryRefiner:
        payload = torch.load(fit_dir / name, map_location=device, weights_only=False)
        model = PointGeometryRefiner(
            GeometryModelConfig(**payload["model_config"])).to(device)
        model.load_state_dict(payload["model_state"])
        return model.eval()

    center = _load_refiner("center_allval.pt")
    rotation = _load_refiner("rotation_allval.pt")
    cov_payload = torch.load(
        fit_dir / "covariance_allval.pt", map_location=device, weights_only=False)
    covariance = PointConditionedRotationCovariance(
        cov_payload["global_error_variance"]).to(device)
    covariance.load_state_dict(cov_payload["model_state"])
    covariance.eval()
    return GeometryModels(center, rotation, covariance,
                          temperature=float(cov_payload["temperature"]))


def predict_measurements(candidates: list[dict], points_xyzi: np.ndarray,
                         frame_id: int, models: GeometryModels,
                         device: str) -> tuple[DetectionGeometryMeasurement, ...]:
    """Build the typed per-box measurements ``FrozenGeometryRefiner`` expects.

    Center and rotation use different ROI schemas (cap-safe spatial-coverage
    sampling for center, deterministic linspace sampling for rotation) — this
    matches ``geometry_freeze.FROZEN_GEOMETRY["center"]["learned_measurement"]``
    and must not be unified.
    """
    center_frame = build_center_frame_v2(frame_id, candidates, _NO_GT, points_xyzi)
    rotation_frame = build_frame_record(frame_id, candidates, _NO_GT, points_xyzi)

    center_mean = predict_frame(models.center, center_frame, device)[:, :3]
    rotation_mean = predict_frame(models.rotation, rotation_frame, device)[:, 6:9]
    point_variance = models.temperature * predict_frame(
        models.covariance, rotation_frame, device)

    points_xyz = points_xyzi[:, :3]
    measurements = []
    for index, box in enumerate(candidates):
        dims, debug = point_support_dimensions(
            box, points_xyz, FROZEN_REFINEMENT_BASELINE)
        measurements.append(DetectionGeometryMeasurement(
            center_delta_local_m=center_mean[index],
            extent=ExtentMeasurement(dims, debug),
            rotation_delta_local_rad=rotation_mean[index],
            rotation_covariance_tangent_rad2=point_variance[index]))
    return tuple(measurements)


def refine_frame(runtime: FrozenGeometryRefiner, candidates: list[dict],
                 points_xyzi: np.ndarray, frame_id: int, models: GeometryModels,
                 device: str) -> list[dict]:
    """Run one frame through the full measurement + causal-filtering pipeline.

    Create one ``runtime`` per scene (``FrozenGeometryRefiner()``) and call
    this once per frame in strictly increasing ``frame_id`` order; call
    ``runtime.reset()`` between scenes.
    """
    measurements = predict_measurements(candidates, points_xyzi, frame_id, models, device)
    return runtime.process_frame(FrameGeometryMeasurements(
        frame_id, tuple(candidates), measurements))
