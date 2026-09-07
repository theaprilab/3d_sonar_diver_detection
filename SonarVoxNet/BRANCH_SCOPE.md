# Augmentation ablation branch

Branch name: `ablation/augmentation`

This branch must change data augmentation only. The model architecture, target
definition, optimizer, split files, and evaluation configuration remain fixed
to the main recipe.

The sibling `reference_source/` directory contains augmentation, on-the-fly
dataset, and cache-generation candidates. Before migration, convert all paths
to explicit command-line inputs and document whether each augmentation is
materialized in a cache or generated online. Never commit generated caches,
augmented point clouds, or archive files.
