# Head and Backbone Ablation Protocol

This branch compares the detection formulation and the 3D middle encoder. It
does not change the dataset split, voxelization bounds, VFE, BEV backbone,
optimizer, training schedule, score threshold, NMS setting, or evaluation
implementation.

| Recipe | Changed component | Fixed rotation target |
| --- | --- | --- |
| `dense_middle_6d_l1.yaml` | Sparse middle encoder → dense Conv3D middle encoder | Full 3D 6D target with L1 loss |
| `center_zyaw.yaml` | Center head uses a yaw-only target | Yaw projected to 6D with L1 loss |
| `anchor_zyaw.yaml` | Center head → anchor head | Same yaw-projected 6D target |

The anchor recipe uses two BEV templates at every feature-map location. Its
template geometry is stored in the recipe, rather than hidden in Python code.
The regression output contains center residuals, log dimensions, and a 6D
rotation target. The template yaw is used for matching and box placement; the
rotation branch is still trained against the documented yaw-only 6D target.

`models.targets.project_to_zyaw` derives that target from the full annotation
rotation matrix. Therefore a center-versus-anchor comparison does not silently
compare full-3D supervision against yaw-only supervision.

For every result, record the recipe path, random seed, dataset split identifier,
checkpoint selection rule, and exact evaluation command. A changed value must
be reported as a separate experimental condition rather than folded into this
ablation.
