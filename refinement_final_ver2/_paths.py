"""Read-only import bridge to the existing VoxelNet implementation.

Set ``VOXELNET_MODEL_ROOT`` to the checkout's root if this folder is copied
somewhere that isn't a few levels under a directory containing
``VoxelNet/model`` (e.g. shared standalone with a teammate).
"""

from pathlib import Path
import os
import sys


_override = os.environ.get("VOXELNET_MODEL_ROOT")
if _override:
    VOXELNET_MODEL = Path(_override) / "VoxelNet" / "model"
else:
    VOXELNET_MODEL = None
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / "VoxelNet" / "model"
        if candidate.is_dir():
            VOXELNET_MODEL = candidate
            break
    if VOXELNET_MODEL is None:
        raise RuntimeError(
            "Could not locate VoxelNet/model above this file; set "
            "VOXELNET_MODEL_ROOT to the checkout root that contains it.")

if str(VOXELNET_MODEL) not in sys.path:
    sys.path.insert(0, str(VOXELNET_MODEL))

