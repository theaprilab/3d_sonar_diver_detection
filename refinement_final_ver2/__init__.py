"""Causal geometry refinement — full pipeline, shareable module.

A strictly causal, test-time stage that refines the 9-DoF boxes of a 3D
detector by filtering each geometry axis (size, center, rotation) over time.
Two learned point-conditioned models produce the per-frame center/rotation
residuals and rotation covariance; a training-free routine produces the size
measurement; a numpy-only causal filtering core turns those measurements into
refined boxes. It never promotes a sub-threshold candidate, so recall stays
bounded by the detector.

Public API
----------
    FrozenGeometryRefiner      -- causal filtering core, single entry point (per scene)
    FrameGeometryMeasurements  -- typed per-frame input
    DetectionGeometryMeasurement, ExtentMeasurement
    apply_residual             -- pure-numpy box residual application
    GeometryModels, load_geometry_models, predict_measurements, refine_frame
                               -- learned measurement layer + end-to-end glue

See ``README.md`` for the pipeline, the measurement contract, and provenance.
Fitted weights (``center_allval.pt`` / ``rotation_allval.pt`` /
``covariance_allval.pt`` per seed) are distributed separately.
"""

from __future__ import annotations

from frozen_geometry_runtime import (
    DetectionGeometryMeasurement,
    ExtentMeasurement,
    FrameGeometryMeasurements,
    FrozenGeometryRefiner,
)
from geometry_freeze import FROZEN_GEOMETRY, FROZEN_GEOMETRY_VERSION
from geometry_ops import apply_residual
from pipeline import (
    GeometryModels,
    load_geometry_models,
    predict_measurements,
    refine_frame,
)

__all__ = [
    "FrozenGeometryRefiner",
    "FrameGeometryMeasurements",
    "DetectionGeometryMeasurement",
    "ExtentMeasurement",
    "apply_residual",
    "FROZEN_GEOMETRY",
    "FROZEN_GEOMETRY_VERSION",
    "GeometryModels",
    "load_geometry_models",
    "predict_measurements",
    "refine_frame",
]
