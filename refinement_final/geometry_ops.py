"""Pure-numpy box residual application (no torch dependency).

Extracted from the training/evaluation code so the frozen runtime depends only
on numpy.  Applies a residual to a detector box expressed in the box's own
local frame: translation and log-size in the box axes, rotation on SO(3).
"""

from __future__ import annotations

import copy

import numpy as np

from bayesian_box_filter import so3_exp


def apply_residual(box: dict, residual: np.ndarray, components: str) -> dict:
    """Apply a 9-vector residual (center / log-dims / tangent-rotation) to a box.

    ``components`` selects which parts act: ``"c"`` center (``residual[:3]``, in
    the box's local axes), ``"d"`` log-dimensions (``residual[3:6]``), ``"r"``
    rotation (``residual[6:9]``, right-multiplicative ``SO(3)`` exponential).
    """
    item = copy.deepcopy(box)
    rotation = np.asarray(box["R"], dtype=np.float64)
    if "c" in components:
        center = np.array([box["x"], box["y"], box["z"]], dtype=np.float64)
        center += rotation @ residual[:3]
        item["x"], item["y"], item["z"] = map(float, center)
    if "d" in components:
        dims = np.array([box["l"], box["w"], box["h"]], dtype=np.float64)
        dims *= np.exp(residual[3:6])
        item["l"], item["w"], item["h"] = map(float, dims)
    if "r" in components:
        item["R"] = rotation @ so3_exp(residual[6:9])
    return item
