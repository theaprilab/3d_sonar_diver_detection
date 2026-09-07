# Augmentation ablation branch

Branch name: `ablation/augmentation`

This branch changes data augmentation only. The model architecture, target
definition, optimizer, split files, and evaluation configuration remain fixed
to the main recipe. Every candidate transform, including global yaw rotation
and front/back reflection, is recorded explicitly so its empirical and physical
validity can be assessed before a final recipe is selected.

`src/sonarvoxnet/data/augmentation.py` contains dataset-independent transforms
with full-3D box updates. Recipes are under `configs/ablations/augmentation/`.
The data adapter is responsible for choosing deterministic per-epoch,
per-sample seeds and for documenting whether augmentation is materialized in a
cache or generated online. Never commit generated caches, augmented point
clouds, or archive files.
