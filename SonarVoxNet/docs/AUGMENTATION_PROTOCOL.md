# Augmentation protocol

This branch evaluates augmentation while keeping the model, voxel grid, data
split, optimizer, checkpoint-selection rule, decoder, and evaluator fixed.
Every reported result must identify one YAML recipe in
`configs/ablations/augmentation/`.

## Allowed transforms

The controlled transforms are per-box yaw perturbation and translation,
left/right and front/back reflection, global scale, global translation, global
yaw rotation, and optional GT sampling. Full-3D box rotations are updated
together with their in-box points.

The fixed forward-looking sensor geometry makes global yaw rotation and
front/back reflection physically questionable candidates. They remain available
in this search branch only when their use is stated in the recipe and result
table. The final recipe and any cross-method benchmark protocol may exclude
them after the ablation is evaluated.

## Reproducibility

The augmentation random generator must be derived from `(base_seed, epoch,
sample_index)`, not from worker order or `None`. The run manifest must record
the recipe path, base seed, whether samples are generated online or cached, and
the exact training split including empty frames.

## Controlled comparison

Use the recipes cumulatively: `none`, `per_box`, `per_box_flip`,
`per_box_flip_scale`, and `per_box_flip_scale_translate`. Do not compare an
unlabeled bundle of transforms against no augmentation. GT sampling is reported
as a separate factor (`gt_sampling.yaml`) because it changes object count and
class balance. Candidate GT-sampling locations must be drawn only from the
documented sensor FOV and collision-rejected before insertion.

## Cache policy

Derived caches and GT databases are local artifacts and are ignored by Git.
Validation and test data are never augmented. If training uses empty/background
frames, the online and cached data paths must retain the same empty-frame policy.
