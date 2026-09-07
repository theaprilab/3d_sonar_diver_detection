# Migration boundary

This draft has a deliberately narrow boundary.  The following reviewed
components must be migrated before the first public release:

| Public destination | Source to review | Required release work |
| --- | --- | --- |
| `src/sonarvoxnet/data/` | `VoxelNet/model/voxelize.py`, cache reader, label preparation | Remove laboratory paths; document input schema and cache version. |
| `src/sonarvoxnet/targets/` | Center heatmap and 3D rotation target generation | Keep full-3D and z-yaw target modes explicit and tested. |
| `src/sonarvoxnet/ablations/anchor.py` | `VoxelNet/pointpillars_head_kit/aprilab_heads/` | Translate comments to English and preserve the head-only comparison contract. |
| `src/sonarvoxnet/training/` | Reviewed center-head training loop | Replace feature flags with a resolved config and persist run metadata. |
| `src/sonarvoxnet/evaluation/` | 3D OBB evaluator and extra rotation metrics | Emit one documented JSON schema for all paper tables. |
| `scripts/reproduce_*.sh` | Final experiment records | Add one exact command per reported table row. |

The following exploratory components are intentionally out of scope unless a
final paper claim requires them: polar representations, foreground gating,
density auxiliary heads, stage-two refinement, range-conditioned losses,
z-fine encoders, and internal diagnostic scripts.

No code should be copied solely for backward compatibility with old checkpoints.
If a released checkpoint needs compatibility code, pin it to a release tag and
test it as part of the public evaluation command.
