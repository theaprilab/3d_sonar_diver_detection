# Rotation ablation branch

Branch name: `ablation/rotation`

This branch compares quaternion, axis-angle, 6D, and 9D rotation
representations. The VFE, sparse middle encoder, BEV backbone, center head,
data split, and non-rotation training settings must remain equal to `main`.

Included recipes:

- `rotation_quaternion.yaml`
- `rotation_axis_angle.yaml`
- `rotation_9d.yaml`

The sibling `reference_source/` directory contains the existing rotation
conversion, target generation, loss, and diagnostic/evaluation candidates.
Migrate only the reviewed representation-specific logic and full-SO(3) metrics;
exclude experiment logs and deprecated comparison scripts.
