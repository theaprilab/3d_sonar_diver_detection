"""Stable production facade for the frozen geometry refinement path.

The accepted algorithms remain implemented by the original frozen modules.
This facade only makes measurement semantics and state ownership explicit; it
must stay frame-by-frame equivalent to the Stage B-R ``imm_point`` path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
from typing import Any

import numpy as np

from bayesian_box_filter import so3_exp
from causal_bayesian_center import FrozenCausalOnlineSuite
from causal_rotation_imm import (CausalRotationIMMTrack, ROTATION_IMM_CONFIG,
                                 RotationIMMConfig)
from geometry_ops import apply_residual
from frozen_guard import verify_frozen_sources
from geometry_freeze import FROZEN_GEOMETRY, FROZEN_GEOMETRY_VERSION


RUNTIME_VERSION = "frozen-geometry-runtime-equivalent-stage-b-r-1"
CENTER_GAIN = 0.5


def _vector3(value: Any, name: str, positive: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite length-3 vector")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return array


def _covariance3(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (3,):
        array = np.diag(array)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 3-vector or 3x3 matrix")
    array = 0.5 * (array + array.T)
    if np.min(np.linalg.eigvalsh(array)) <= 0.0:
        raise ValueError(f"{name} must be positive definite")
    return array


@dataclass(frozen=True)
class ExtentMeasurement:
    """Training-free point-supported dimensions for one detector candidate."""

    dimensions_m: Any
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def validated(self) -> tuple[np.ndarray, dict[str, Any]]:
        return (_vector3(self.dimensions_m, "extent dimensions", positive=True),
                copy.deepcopy(self.diagnostics))


@dataclass(frozen=True)
class DetectionGeometryMeasurement:
    """Typed measurements aligned with exactly one frozen detector box.

    ``center_delta_local_m`` is a residual mean, not a temporal covariance.
    ``rotation_covariance_tangent_rad2`` is the actual SO(3)-IMM measurement
    covariance.  Keeping these types separate prevents covariance semantic
    reuse across the two filters.
    """

    center_delta_local_m: Any
    extent: ExtentMeasurement
    rotation_delta_local_rad: Any
    rotation_covariance_tangent_rad2: Any

    def validated(self) -> tuple[np.ndarray, tuple[np.ndarray, dict[str, Any]],
                                 np.ndarray, np.ndarray]:
        return (
            _vector3(self.center_delta_local_m, "center local residual"),
            self.extent.validated(),
            _vector3(self.rotation_delta_local_rad, "rotation tangent residual"),
            _covariance3(self.rotation_covariance_tangent_rad2,
                         "rotation tangent covariance"),
        )


@dataclass(frozen=True)
class FrameGeometryMeasurements:
    frame_id: int
    detections: tuple[dict[str, Any], ...]
    measurements: tuple[DetectionGeometryMeasurement, ...]

    def validate(self) -> None:
        if len(self.detections) != len(self.measurements):
            raise ValueError("detections and geometry measurements must align one-to-one")
        if not isinstance(self.frame_id, (int, np.integer)):
            raise ValueError("frame_id must be an integer")
        for measurement in self.measurements:
            measurement.validated()


class FrozenGeometryRefiner:
    """Single causal entrypoint preserving the accepted Stage B-R algorithm."""

    method_version = RUNTIME_VERSION
    freeze_version = FROZEN_GEOMETRY_VERSION

    def __init__(self, rotation_config: RotationIMMConfig = ROTATION_IMM_CONFIG):
        self.integrity = verify_frozen_sources()
        self.rotation_config = rotation_config
        self.reset()

    def reset(self) -> None:
        self.box_suite = FrozenCausalOnlineSuite()
        self.rotation_tracks: dict[int, CausalRotationIMMTrack] = {}
        self.last_rotation_diagnostics: dict[int, dict[str, Any]] = {}
        self.last_frame_id: int | None = None

    def process_frame(self, frame: FrameGeometryMeasurements) -> list[dict[str, Any]]:
        frame.validate()
        if self.last_frame_id is not None and int(frame.frame_id) <= self.last_frame_id:
            raise ValueError("frame ids must be strictly increasing within a scene")
        self.last_frame_id = int(frame.frame_id)

        candidates = [copy.deepcopy(box) for box in frame.detections]
        validated = [measurement.validated() for measurement in frame.measurements]
        corrected, point_measurements = [], []
        rotation_measurements, rotation_covariances = [], []
        for index, (box, values) in enumerate(zip(candidates, validated)):
            center_delta, extent, rotation_delta, rotation_covariance = values
            box["_stage_g_index"] = index
            corrected.append(apply_residual(
                box, np.r_[CENTER_GAIN * center_delta, np.zeros(6)], "c"))
            point_measurements.append(extent)
            rotation_measurements.append(
                np.asarray(box["R"], dtype=np.float64) @ so3_exp(rotation_delta))
            rotation_covariances.append(rotation_covariance)

        base = self.box_suite.process_frame(
            corrected, points=None, frame_id=int(frame.frame_id),
            point_measurements=point_measurements)
        output = []
        for source in base:
            box = dict(source)
            source_index = int(box["_stage_g_index"])
            track_id = int(box["tf_track_id"])
            measurement = rotation_measurements[source_index]
            if track_id not in self.rotation_tracks:
                self.rotation_tracks[track_id] = CausalRotationIMMTrack(
                    measurement, self.rotation_config)
            rotation, diagnostic = self.rotation_tracks[track_id].update(
                int(frame.frame_id), measurement,
                rotation_covariances[source_index])
            box["R"] = rotation
            self.last_rotation_diagnostics[track_id] = diagnostic
            output.append(box)

        active_ids = {int(track.track_id) for track in self.box_suite.box_refiner.tracks}
        self.rotation_tracks = {
            track_id: track for track_id, track in self.rotation_tracks.items()
            if track_id in active_ids
        }
        self.last_rotation_diagnostics = {
            track_id: diagnostic
            for track_id, diagnostic in self.last_rotation_diagnostics.items()
            if track_id in active_ids
        }
        return output

    def provenance(self) -> dict[str, Any]:
        return {
            "runtime_version": self.method_version,
            "frozen_source_integrity": copy.deepcopy(self.integrity),
            "geometry_freeze": copy.deepcopy(FROZEN_GEOMETRY),
            "center_gain": CENTER_GAIN,
            "rotation_imm_config": dict(self.rotation_config.__dict__),
            "causal": True,
        }
