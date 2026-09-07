# Head and backbone ablation branch

Branch name: `ablation/head-backbone`

This branch compares only the middle encoder and detection-head choices while
holding data splits, voxel grid, optimizer recipe, decoding thresholds, and
evaluation protocol fixed.

Included recipes:

- `dense_middle_6d_l1.yaml`
- `anchor_zyaw.yaml`
- `center_zyaw.yaml`

Implementation candidates are under the sibling `reference_source/` directory:
the PointPillars head kit, anchor assignment and decoding utilities, dense and
sparse middle encoders, and the reviewed target/loss files. Migrate them into
`src/sonarvoxnet/models/`, `targets/`, and `training/` with English comments
and focused tests. Do not add raw caches, checkpoints, launch logs, or internal
analysis scripts.
