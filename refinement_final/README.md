# Causal Geometry Refinement

A strictly **causal**, test-time module that refines the 9-DoF oriented boxes of
a 3D detector by filtering each geometry axis over time. It treats the detector
as a sensor and maintains a per-track estimate of box geometry using **only past
and present frames** — never future ones — so it is admissible for real-time,
online use.

It is a *secondary* refinement stage, not a detector: it never promotes a
sub-threshold candidate into a detection, so **recall stays bounded by the
detector** and the stage acts on geometry (translation, scale, orientation) of
boxes the detector already produced.

This directory is a clean, self-contained copy of the frozen module. It is
**numpy-only** (no PyTorch); the learned components are shipped as small fitted
tensors and applied outside this core (see *Measurement contract* below).

---

## Pipeline

Detections are associated into short tracks by bird's-eye-view overlap. Each
track carries a state `x = (size, center ∈ R³, rotation ∈ SO(3))`, and each axis
is filtered on its own geometry:

| Axis | Per-frame measurement | Temporal filter |
|------|-----------------------|-----------------|
| **Size** | conservative, shrink-only point-support extent + temporal log-size average (corrects systematic oversize only) | — (quasi-static; no velocity) |
| **Center** | learned local residual (applied at gain 0.5) | Euclidean interacting-multiple-model (IMM): stationary / constant-velocity / maneuver; depth as a random walk; measurement covariance fixed at the sensor voxel-quantization floor |
| **Rotation** | learned tangent residual + learned point-conditioned tangent covariance | error-state `SO(3)` IMM in the tangent space (right-multiplicative errors `R ← R·exp(ξ)`): stationary / constant-angular-velocity / maneuver |

A per-track existence log-odds temporally smooths each box's confidence. The
stage ends with a final rotated NMS. The **only learned parts** are the center
residual, the rotation residual, and the rotation covariance; everything else is
classical robust estimation. The learned parts are fit on the validation split
under a frozen recipe, with no test-time selection.

The exact, immutable method manifest is `geometry_freeze.py`
(`FROZEN_GEOMETRY`), and the sources are checksum-guarded (see *Integrity*).

---

## Files

```
frozen_geometry_runtime.py   Entry point: FrozenGeometryRefiner + typed inputs
causal_bayesian_center.py    Euclidean center IMM (Kalman-conditional, mode-marginalized)
causal_rotation_imm.py       SO(3) tangent-space error-state IMM
training_free_filter.py      Neutral score, shrink-only size, track association, existence
bayesian_box_filter.py       SO(3) utilities (exp/log, projection, innovation, covariance)
geometry_ops.py              apply_residual (pure-numpy box residual application)
geometry_freeze.py           FROZEN_GEOMETRY manifest (metadata, not an implementation)
frozen_guard.py              Fail-closed SHA-256 integrity check of the core sources
frozen_geometry_checksums.json  Integrity manifest for this directory
__init__.py                  Public API re-exports
```

Requirements: Python 3.10+, `numpy`. No other runtime dependency.

---

## Usage

The module uses flat imports, so add this directory to the path (or run from
inside it):

```python
import sys; sys.path.insert(0, "refinement_final")
import numpy as np
from frozen_geometry_runtime import (
    FrozenGeometryRefiner, FrameGeometryMeasurements,
    DetectionGeometryMeasurement, ExtentMeasurement,
)

refiner = FrozenGeometryRefiner()          # runs the integrity guard

# One scene, replayed frame by frame in acquisition order:
for frame_id, detections, measurements in scene_stream:
    refined_boxes = refiner.process_frame(FrameGeometryMeasurements(
        frame_id=frame_id,
        detections=detections,             # tuple of detector box dicts
        measurements=measurements,          # tuple aligned 1:1 with detections
    ))
```

Create one `FrozenGeometryRefiner` per scene (call `refiner.reset()` between
scenes). Frame ids must be strictly increasing within a scene.

### Measurement contract

Each **detector box** is a dict with keys `x, y, z` (center, metres),
`l, w, h` (dimensions, metres), `R` (3×3 rotation matrix, world frame), and
`score`.

Each **`DetectionGeometryMeasurement`** (aligned one-to-one with a box) carries
the per-frame measurements that the detector-coupled learned models produce:

- `center_delta_local_m` — `(3,)` center residual in the box's local axes
- `extent` — `ExtentMeasurement(dimensions_m=(3,))`, the conservative
  point-support size measurement
- `rotation_delta_local_rad` — `(3,)` rotation tangent residual
- `rotation_covariance_tangent_rad2` — `(3,)` or `(3, 3)` tangent covariance

`process_frame` returns a list of refined box dicts (same schema, plus a
`tf_track_id` and filter diagnostics).

> The learned models that *produce* these measurements are detector-coupled
> (they read the point cloud and the detector features) and are **not** part of
> this numpy core. Their definitions and fitted weights live with the detector
> pipeline; this module is the estimation core that consumes their output.

---

## What it does not do

- It does **not** recover missed divers: no sub-threshold candidate is promoted,
  so the detected-object set is exactly the detector's.
- It is **not** a localization-quality rescorer and does not attempt to inflate
  precision; the only score change is a temporal existence log-odds.
- It is a **filter, not a smoother**: it forms `p(x_t | z_{1:t})`, never using
  future frames, trading the smoothing optimum for strict causality.

---

## Integrity & provenance

`frozen_guard.verify_frozen_sources()` (invoked in the constructor) recomputes
the SHA-256 of every algorithm file against `frozen_geometry_checksums.json` and
**raises** on any mismatch. This pins the shared copy to an exact source state.

`geometry_freeze.py` records the frozen method verbatim: pipeline order, the
center residual gain (0.5), and the per-axis filter choices. It was selected on
validation (leave-one-scene-out) **before** the final fit; the test split was
not read during selection.
