"""Deterministic seeds for per-epoch, per-sample augmentation."""

from __future__ import annotations

import numpy as np


def augmentation_rng(base_seed: int, epoch: int, sample_index: int) -> np.random.Generator:
    """Return a reproducible generator without relying on worker scheduling."""
    sequence = np.random.SeedSequence([base_seed, epoch, sample_index])
    return np.random.default_rng(sequence)
