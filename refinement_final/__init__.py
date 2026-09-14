"""Causal geometry refinement — frozen, shareable module.

A strictly causal, test-time stage that refines the 9-DoF boxes of a 3D
detector by filtering each geometry axis (size, center, rotation) over time.
It consumes per-frame measurements and returns refined boxes; it never
promotes a sub-threshold candidate, so recall stays bounded by the detector.

Public API
----------
    FrozenGeometryRefiner      -- single causal entry point (per scene)
    FrameGeometryMeasurements  -- typed per-frame input
    DetectionGeometryMeasurement, ExtentMeasurement
    apply_residual             -- pure-numpy box residual application

See ``README.md`` for the pipeline, the measurement contract, and provenance.
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

__all__ = [
    "FrozenGeometryRefiner",
    "FrameGeometryMeasurements",
    "DetectionGeometryMeasurement",
    "ExtentMeasurement",
    "apply_residual",
    "FROZEN_GEOMETRY",
    "FROZEN_GEOMETRY_VERSION",
]
