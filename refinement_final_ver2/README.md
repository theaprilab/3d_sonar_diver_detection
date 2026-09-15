# Causal Geometry Refinement — Full Pipeline

A strictly **causal**, test-time module that refines the 9-DoF oriented boxes of
a 3D detector by filtering each geometry axis over time. It treats the detector
as a sensor and maintains a per-track estimate of box geometry using **only past
and present frames** — never future ones — so it is admissible for real-time,
online use.

It is a *secondary* refinement stage, not a detector: it never promotes a
sub-threshold candidate into a detection, so **recall stays bounded by the
detector** and the stage acts on geometry (translation, scale, orientation) of
boxes the detector already produced.

This is the full pipeline, including the learned models that produce the
per-frame measurements — the previous shared copy (`refinement_final/`) shipped
only the numpy-only causal filtering core and described the learned models as
living "with the detector pipeline," which was misleading: they are a separate,
detector-coupled but architecturally independent stage that lives here, not
inside the detector.

---

## Two layers

| Layer | What it does | Depends on |
|---|---|---|
| **Learned measurement models** | Given a detector box and the frame's raw sonar points, predict a center residual, a rotation residual, and a rotation covariance | PyTorch, fitted weights (distributed separately) |
| **Causal filtering core** | Given those measurements plus a training-free size estimate, track each box over time and output the refined geometry | numpy only |

`pipeline.py` is the glue between them.

### Learned measurement models

| Axis | Per-frame measurement | ROI schema |
|------|-----------------------|------------|
| **Center** | `PointGeometryRefiner` residual, `[:3]` slice, applied at gain 0.5 | `learned_center_data_v2.build_center_frame_v2` (cap-safe spatial-coverage point sampling) |
| **Rotation** | `PointGeometryRefiner` residual, `[6:9]` slice (tangent) | `learned_data.build_frame_record` (deterministic linspace point sampling) |
| **Rotation covariance** | `PointConditionedRotationCovariance`, scaled by a fitted `temperature` | same ROI as rotation |
| **Size** | `training_free_filter.point_support_dimensions` (training-free, shrink-only point support) | raw xyz points, no ROI/model |

Center and rotation use two *different* ROI-building schemas — this is not
duplication to clean up, it's the frozen recipe
(`geometry_freeze.FROZEN_GEOMETRY["center"]["learned_measurement"]` records the
V2 schema explicitly). Both feed the same `PointGeometryRefiner` architecture
(it has independently fitted weights per axis) plus a separate
`PointConditionedRotationCovariance` for the tangent covariance.

### Causal filtering core

Detections are associated into short tracks by bird's-eye-view overlap. Each
track carries a state `x = (size, center ∈ R³, rotation ∈ SO(3))`, and each axis
is filtered on its own geometry:

| Axis | Temporal filter |
|------|-----------------|
| **Size** | quasi-static; temporal log-size average only, no velocity |
| **Center** | Euclidean interacting-multiple-model (IMM): stationary / constant-velocity / maneuver; depth as a random walk; measurement covariance fixed at the sensor voxel-quantization floor |
| **Rotation** | error-state `SO(3)` IMM in the tangent space (right-multiplicative errors `R ← R·exp(ξ)`): stationary / constant-angular-velocity / maneuver |

A per-track existence log-odds temporally smooths each box's confidence. The
stage ends with a final rotated NMS.

The exact, immutable method manifest for the filtering core is
`geometry_freeze.py` (`FROZEN_GEOMETRY`), and its sources are checksum-guarded
(see *Integrity*, below — this covers the filtering core only, not the learned
measurement models).

---

## Files

```
Filtering core (numpy only, checksum-guarded):
  frozen_geometry_runtime.py   Entry point: FrozenGeometryRefiner + typed inputs
  causal_bayesian_center.py    Euclidean center IMM (Kalman-conditional, mode-marginalized)
  causal_rotation_imm.py       SO(3) tangent-space error-state IMM
  training_free_filter.py      Neutral score, shrink-only size, track association, existence
  bayesian_box_filter.py       SO(3) utilities (exp/log, projection, innovation, covariance)
  geometry_ops.py              apply_residual (pure-numpy box residual application)
  geometry_freeze.py           FROZEN_GEOMETRY manifest (metadata, not an implementation)
  frozen_guard.py              Fail-closed SHA-256 integrity check of the core sources
  frozen_geometry_checksums.json  Integrity manifest for the filtering core

Learned measurement models (PyTorch):
  learned_geometry_model.py     PointGeometryRefiner (center + rotation residual net)
  learned_rotation_uncertainty.py  PointConditionedRotationCovariance
  learned_data.py                object_roi / build_frame_record (rotation ROI schema)
                                  + shared feature-dim constants + detection<->GT matching
  learned_center_data_v2.py      object_roi_center_v2 / build_center_frame_v2 (center ROI schema)
  predict.py                     predict_frame: forward pass over a built ROI batch

Glue:
  pipeline.py                   load_geometry_models / predict_measurements / refine_frame
  _paths.py                     import bridge to VoxelNet/model (for OBB math only)
  __init__.py                   public API re-exports
```

Requirements: Python 3.10+, `numpy`, `torch`, `scipy` (Hungarian assignment in
`learned_data.match_detections`). `learned_data.py` and `learned_center_data_v2.py`
also import `eval_voxelnet` from the detector's `VoxelNet/model/` for pure OBB
math (`pred_obb_from_box`, `gt_obb_from_row`, `iou_3d_obb`) — not the detector's
network itself, but importing that module pulls in `torch`/`shapely` and the
rest of `VoxelNet/model` as a side effect. `_paths.py` locates it automatically
by walking up from this file; set `VOXELNET_MODEL_ROOT` to override if this
folder is copied somewhere that isn't a few levels under a checkout that has
`VoxelNet/model`.

**Fitted weights are not included.** `center_allval.pt`, `rotation_allval.pt`,
and `covariance_allval.pt` (per seed) are distributed separately.

---

## Usage

The module uses flat imports, so add this directory to the path (or run from
inside it):

```python
import sys; sys.path.insert(0, "refinement_final_ver2")
import numpy as np
from pathlib import Path
from pipeline import load_geometry_models, refine_frame
from frozen_geometry_runtime import FrozenGeometryRefiner

device = "cuda"
models = load_geometry_models(Path("artifacts/seed0"), device)  # the 3 .pt files above

for scene in scenes:
    runtime = FrozenGeometryRefiner()          # one per scene; runs the integrity guard
    for frame_id, detections, points_xyzi in scene:
        # detections: list of detector box dicts (see contract below)
        # points_xyzi: (N, 4) raw retained sonar points [x, y, z, intensity], world frame
        refined_boxes = refine_frame(
            runtime, detections, points_xyzi, frame_id, models, device)
```

Frame ids must be strictly increasing within a scene. If you already have the
per-frame measurements computed elsewhere, call `FrozenGeometryRefiner.process_frame`
directly with a `FrameGeometryMeasurements` instead of going through `pipeline.py`
— see `predict_measurements` in `pipeline.py` for the exact construction.

### Detector box contract

Each **detector box** is a dict with keys `x, y, z` (center, metres),
`l, w, h` (dimensions, metres), `R` (3×3 rotation matrix, world frame), and
`score`.

### Measurement contract (what `pipeline.predict_measurements` builds)

- `center_delta_local_m` — `(3,)` center residual in the box's local axes
- `extent` — `ExtentMeasurement(dimensions_m=(3,))`, the conservative
  point-support size measurement
- `rotation_delta_local_rad` — `(3,)` rotation tangent residual
- `rotation_covariance_tangent_rad2` — `(3,)` tangent covariance

`refine_frame` returns a list of refined box dicts (same schema, plus a
`tf_track_id` and filter diagnostics).

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

`frozen_guard.verify_frozen_sources()` (invoked in `FrozenGeometryRefiner`'s
constructor) recomputes the SHA-256 of every **filtering-core** source file
against `frozen_geometry_checksums.json` and **raises** on any mismatch. This
pins the shared copy of the classical filtering algorithm to an exact source
state. It does not cover the learned measurement models — those are pinned by
their fitted weights instead (a changed architecture simply fails to load a
mismatched `state_dict`).

`geometry_freeze.py` records the frozen method verbatim: pipeline order, the
center residual gain (0.5), the per-axis filter choices, and which ROI schema
each learned measurement uses. It was selected on validation (leave-one-scene-out)
**before** the final fit; the test split was not read during selection.
