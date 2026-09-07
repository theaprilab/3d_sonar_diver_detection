# SonarVoxNet Internal Structure Guide

This document is for internal team communication only. It is outside the
public release directory, so it will not be included when only
`3d_sonar_diver_detection/SonarVoxNet/` is added to Git.

## Location

The publication-oriented code draft is located at:

```text
3d_sonar_diver_detection/SonarVoxNet/
```

The `SonarVoxNet/` directory at the workspace root is the independent draft
used to create this layout. The copy under `3d_sonar_diver_detection/` is the
current candidate for the existing repository.

## Directories and files

```text
SonarVoxNet/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── configs/
├── docs/
├── src/sonarvoxnet/
├── tools/
└── tests/
```

- `README.md`: Describes the model scope, the fixed main model, ablations,
  installation, and the intended command interface.
- `LICENSE`: A temporary notice that this is a pre-release draft. Replace it
  with the selected open-source license before public release.
- `.gitignore`: Excludes data, caches, checkpoints, logs, and Python caches
  from the release.
- `pyproject.toml`: Defines Python package metadata and the base, sparse, and
  development dependency groups.

### `configs/`

This directory fixes every paper experiment as a YAML recipe. A configuration
file, rather than an undocumented command-line override, should represent one
reported experimental condition.

- `main_spconv_6d_l1.yaml`: The fixed model: sparse middle encoder, center
  head, full-3D 6D rotation, and L1 target loss.
- `ablations/dense_middle_6d_l1.yaml`: Dense Conv3D middle-encoder comparison.
- `ablations/anchor_zyaw.yaml`: Anchor-head and z-yaw target comparison.
- `ablations/center_zyaw.yaml`: Center-head and z-yaw target comparison.
- `ablations/rotation_quaternion.yaml`: Quaternion rotation-representation
  comparison.
- `ablations/rotation_axis_angle.yaml`: Axis-angle rotation-representation
  comparison.
- `ablations/rotation_9d.yaml`: Raw 9D rotation-matrix representation with
  MSE target loss.

Each ablation recipe inherits the main recipe through `inherits` and changes
only the relevant setting.

### `src/sonarvoxnet/`

This is the Python library. It must not contain laboratory-specific absolute
paths.

- `config.py`: Loads YAML configurations and resolves recursive `inherits`
  fields.
- `models/`: Network architecture code.
  - `model.py`: VFE, dense and sparse middle encoders, BEV backbone, center
    head, and the `SonarVoxNet` model.
  - `rotation.py`: SO(3) decoders for 6D, quaternion, axis-angle, and 9D
    representations.
- `training/`: Shared training code.
  - `losses.py`: The default 6D L1 target loss and the MSE target loss for the
    9D ablation.
- `evaluation/`: Shared metrics and evaluation protocol materials.
  - `metrics.py`: Full SO(3) geodesic rotation error.
  - `protocol/README.md`: The intended location for the final 3D OBB IoU, AP,
    prediction JSON schema, split, and threshold rules.

### `tools/`

This directory contains the user-facing CLI entry points. They parse arguments
and configurations and start an action; reusable implementation belongs in
`src/sonarvoxnet/`.

- `prepare_data.py`: Reserved command for converting raw data to the public
  cache format.
- `train.py`: Reserved command for configuration-based training.
- `evaluate.py`: Reserved command for checkpoint evaluation and result-JSON
  generation.
- `show_config.py`: Prints a fully resolved YAML configuration.

`prepare_data.py`, `train.py`, and `evaluate.py` currently print a migration
notice. This prevents unpublished exploratory code and hidden paths from being
silently copied into the public release.

### `docs/`

- `DATASET.md`: To document dataset access, licensing, raw layout, annotation
  convention, splits, and preprocessing before release.
- `REPRODUCIBILITY.md`: Release checklist for checkpoint metadata, seeds,
  environment versions, and paper-table-to-command mappings.
- `MIGRATION.md`: Maps reviewed components in the existing `VoxelNet` code to
  their release destinations and records excluded exploratory components.

### `tests/`

Fast unit tests.

- `test_config.py`: Checks main and ablation configuration inheritance.
- `test_rotation.py`: Checks 6D rotation encoding/decoding and SO(3)
  constraints.
- `test_metrics.py`: Checks that identical rotations have zero geodesic error.

## Items currently excluded

Raw data, caches, checkpoints, training logs, internal analyses, and
exploratory polar, foreground-gating, and stage-two branches are not part of
the currently fixed paper scope.
