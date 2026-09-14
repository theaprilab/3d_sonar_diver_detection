# Final model + refiner artifacts (local mirror)

Fetched 2026-09-14 from Modal volume `voxelnet-data`. These are the exact weights
behind the `CONFIRMATORY_REPORT.md` test_full numbers.

## Contents

```
detector/
  voxelnet_otfsp_stdsafe_asp_emp_gtcurr65floor1_rngv2_rawbest_final_s0_best_iou35.pt   (25M)
  voxelnet_otfsp_stdsafe_asp_emp_gtcurr65floor1_rngv2_rawbest_final_s1_best_iou35.pt   (25M)
seed0/ , seed1/
  center_allval.pt        learned Center V2 residual regressor (all-val fit)
  rotation_allval.pt      learned point-conditioned rotation tangent residual (all-val fit)
  covariance_allval.pt    point-conditioned tangent covariance + temperature
  final_fit_manifest.json per-seed fit metadata
```

## Detector (final aug recipe)

- Baseline name: `voxelnet_otfsp_stdsafe_asp_emp_gtcurr65floor1_rngv2_rawbest_final`
- Checkpoint selected by best val AP3D@0.35 (`_best_iou35.pt`).
- Volume source: `checkpoints/{baseline}_s{seed}_best_iou35.pt`

## Refiner (frozen geometry suite fit on top of the detector)

- The three `*_allval.pt` per seed are the **only learned parts** of the refinement
  stage. Fit on all 7 validation scenes under the frozen recipe (epoch 10,
  cosine T_max=20), **no test-time selection**.
- Per-seed fit: seed0 temperature **0.9878**, seed1 temperature **1.0341**;
  6055 / 6138 residual pairs; `test_split_read: false`.
- Volume source: `refinement_v3/baselines/{baseline}/final_eval/seed{seed}/`

## Structural (non-fit) hyperparameters — not weights

These live in code, not in the `.pt` files, and define how the learned pieces
are wired into the classical filters:

- `../../geometry_freeze.py`  — FROZEN_GEOMETRY manifest (pipeline order, center
  gain 0.5, per-axis filter choices). Frozen structural hyperparameters.
- `../../frozen_geometry_runtime.py` — CENTER_GAIN = 0.5.
- `../../causal_rotation_imm.py` — ROTATION_IMM_CONFIG (mode_persistence 0.90,
  process_scales stationary/CAV/maneuver = 0.25/1.0/16.0).
- `../../causal_bayesian_center.py` — center IMM (stationary/CV/maneuver,
  mode_stay 0.99, sensor-floor covariance).
- `../../training_free_filter.py` — FROZEN_REFINEMENT_BASELINE (neutral score +
  robust bidirectional shrink-only size).

## To reproduce the test_full apply

`final_eval/modal_final_apply.py` loads `{center,rotation,covariance}_allval.pt`
from the volume, replays `FrozenGeometryRefiner`, and scores with the canonical
protocol (score 0.30 / NMS 0.10 / 3D-IoU 0.30 TP).
