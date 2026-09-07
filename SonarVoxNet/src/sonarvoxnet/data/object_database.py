"""Geometry-only primitives for deterministic GT sampling.

The dataset adapter supplies valid candidate boxes within its sensor field of
view. This module deliberately does not invent a dataset-specific range or FOV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sonarvoxnet.data.augmentation import Box3D, boxes_collide_bev, points_in_box


@dataclass(frozen=True)
class ObjectDatabaseEntry:
    """One object's points expressed in the object's local coordinate frame."""

    local_points: np.ndarray
    dimensions: np.ndarray
    label: str


def build_object_database(samples: list[tuple[np.ndarray, list[Box3D]]], minimum_points: int = 5) -> list[ObjectDatabaseEntry]:
    """Extract valid object crops from training samples only."""
    entries: list[ObjectDatabaseEntry] = []
    for points, boxes in samples:
        for box in boxes:
            crop = points[points_in_box(points, box)].copy()
            if len(crop) < minimum_points:
                continue
            crop[:, :3] = (crop[:, :3] - box.center) @ box.rotation
            entries.append(ObjectDatabaseEntry(crop, box.dimensions.copy(), box.label))
    return entries


def insert_object(points: np.ndarray, boxes: list[Box3D], entry: ObjectDatabaseEntry,
                  candidate: Box3D) -> tuple[np.ndarray, list[Box3D]]:
    """Insert one crop when its candidate does not collide in BEV.

    The caller samples ``candidate`` from the documented dataset-specific FOV.
    A rejected candidate returns the input sample unchanged.
    """
    if candidate.label != entry.label or any(boxes_collide_bev(candidate, box) for box in boxes):
        return points, boxes
    crop = entry.local_points.copy()
    crop[:, :3] = crop[:, :3] @ candidate.rotation.T + candidate.center
    return np.concatenate((points, crop), axis=0), [*boxes, candidate]
