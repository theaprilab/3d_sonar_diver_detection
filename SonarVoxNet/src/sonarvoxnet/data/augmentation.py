"""Deterministic, geometry-consistent point-cloud augmentation.

All functions operate on a point array ``(N, >=3)`` and full-3D oriented boxes.
The branch exposes every candidate transform explicitly; experimental recipes
must record whether a transform is physically valid for the deployment sensor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Box3D:
    """An oriented box with ``world = center + rotation @ local``."""

    center: np.ndarray
    dimensions: np.ndarray
    rotation: np.ndarray
    label: str = "diver"

    def __post_init__(self) -> None:
        if np.asarray(self.center).shape != (3,) or np.asarray(self.dimensions).shape != (3,):
            raise ValueError("center and dimensions must have shape (3,).")
        if np.asarray(self.rotation).shape != (3, 3):
            raise ValueError("rotation must have shape (3, 3).")
        if np.any(np.asarray(self.dimensions) <= 0):
            raise ValueError("box dimensions must be positive.")


@dataclass(frozen=True)
class AugmentationRecipe:
    """Named transforms for one controlled augmentation experiment."""

    per_box_rotation_degrees: float = 0.0
    per_box_translation_std: float = 0.0
    flip_x_probability: float = 0.0
    flip_y_probability: float = 0.0
    scale_range: tuple[float, float] = (1.0, 1.0)
    translation_std: tuple[float, float, float] = (0.0, 0.0, 0.0)
    global_yaw_degrees: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.flip_x_probability <= 1.0 or not 0.0 <= self.flip_y_probability <= 1.0:
            raise ValueError("flip probabilities must be in [0, 1].")
        if self.scale_range[0] <= 0 or self.scale_range[0] > self.scale_range[1]:
            raise ValueError("scale_range must be positive and ordered.")
        if self.per_box_rotation_degrees < 0 or self.per_box_translation_std < 0 or self.global_yaw_degrees < 0:
            raise ValueError("augmentation magnitudes must be non-negative.")


def _rotation_z(angle_radians: float) -> np.ndarray:
    cosine, sine = np.cos(angle_radians), np.sin(angle_radians)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def points_in_box(points: np.ndarray, box: Box3D) -> np.ndarray:
    """Return a mask for points contained in a full-3D oriented box."""
    local = (points[:, :3] - box.center) @ box.rotation
    return np.all(np.abs(local) <= box.dimensions * 0.5, axis=1)


def _box_bev_corners(box: Box3D) -> np.ndarray:
    half_length, half_width = box.dimensions[:2] * 0.5
    local = np.array(
        [[-half_length, -half_width, 0.0], [-half_length, half_width, 0.0],
         [half_length, half_width, 0.0], [half_length, -half_width, 0.0]]
    )
    return (local @ box.rotation.T + box.center)[:, :2]


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    return (_orientation(a, b, c) * _orientation(a, b, d) <= 0 and
            _orientation(c, d, a) * _orientation(c, d, b) <= 0)


def _point_in_convex_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    signs = [_orientation(polygon[index], polygon[(index + 1) % len(polygon)], point) for index in range(len(polygon))]
    return min(signs) >= 0 or max(signs) <= 0


def boxes_collide_bev(first: Box3D, second: Box3D) -> bool:
    """Conservative BEV collision check used before accepting a box perturbation."""
    first_polygon, second_polygon = _box_bev_corners(first), _box_bev_corners(second)
    for polygon_a, polygon_b in ((first_polygon, second_polygon), (second_polygon, first_polygon)):
        if any(_point_in_convex_polygon(point, polygon_b) for point in polygon_a):
            return True
        for index in range(4):
            if _segments_intersect(polygon_a[index], polygon_a[(index + 1) % 4], polygon_b[index], polygon_b[(index + 1) % 4]):
                return True
    return False


def per_box_perturb(points: np.ndarray, boxes: list[Box3D], rng: np.random.Generator,
                    max_rotation_degrees: float, translation_std: float) -> tuple[np.ndarray, list[Box3D]]:
    """Move each object and its original in-box points without changing tilt."""
    result_points = points.copy()
    result_boxes = list(boxes)
    original_masks = [points_in_box(points, box) for box in boxes]
    for index, original in enumerate(boxes):
        angle = np.deg2rad(rng.uniform(-max_rotation_degrees, max_rotation_degrees))
        world_rotation = _rotation_z(angle)
        candidate = replace(
            original,
            center=original.center + rng.normal(0.0, translation_std, size=3),
            rotation=world_rotation @ original.rotation,
        )
        if any(boxes_collide_bev(candidate, other) for other_index, other in enumerate(result_boxes) if other_index != index):
            continue
        local_points = (points[original_masks[index], :3] - original.center) @ original.rotation
        result_points[original_masks[index], :3] = local_points @ candidate.rotation.T + candidate.center
        result_boxes[index] = candidate
    return result_points, result_boxes


def _reflect(points: np.ndarray, boxes: list[Box3D], reflection: np.ndarray) -> tuple[np.ndarray, list[Box3D]]:
    result_points = points.copy()
    result_points[:, :3] = result_points[:, :3] @ reflection.T
    result_boxes = [replace(box, center=reflection @ box.center, rotation=reflection @ box.rotation @ reflection) for box in boxes]
    return result_points, result_boxes


def flip_x(points: np.ndarray, boxes: list[Box3D]) -> tuple[np.ndarray, list[Box3D]]:
    """Reflect the scene around the lateral axis for a controlled ablation."""
    return _reflect(points, boxes, np.diag([-1.0, 1.0, 1.0]))


def flip_y(points: np.ndarray, boxes: list[Box3D]) -> tuple[np.ndarray, list[Box3D]]:
    """Reflect the scene around the forward axis while preserving SO(3) boxes."""
    return _reflect(points, boxes, np.diag([1.0, -1.0, 1.0]))


def global_scale(points: np.ndarray, boxes: list[Box3D], scale: float) -> tuple[np.ndarray, list[Box3D]]:
    """Scale world coordinates, centers, and dimensions by one positive factor."""
    result_points = points.copy()
    result_points[:, :3] *= scale
    return result_points, [replace(box, center=box.center * scale, dimensions=box.dimensions * scale) for box in boxes]


def global_translate(points: np.ndarray, boxes: list[Box3D], translation: np.ndarray) -> tuple[np.ndarray, list[Box3D]]:
    """Translate points and centers by the same world-coordinate vector."""
    result_points = points.copy()
    result_points[:, :3] += translation
    return result_points, [replace(box, center=box.center + translation) for box in boxes]


def global_yaw_rotate(points: np.ndarray, boxes: list[Box3D], angle_radians: float) -> tuple[np.ndarray, list[Box3D]]:
    """Apply one global world-frame yaw rotation for an explicit ablation."""
    rotation = _rotation_z(angle_radians)
    result_points = points.copy()
    result_points[:, :3] = result_points[:, :3] @ rotation.T
    return result_points, [replace(box, center=rotation @ box.center, rotation=rotation @ box.rotation) for box in boxes]


def augment_sample(points: np.ndarray, boxes: list[Box3D], recipe: AugmentationRecipe,
                   rng: np.random.Generator) -> tuple[np.ndarray, list[Box3D]]:
    """Apply one explicit recipe in a fixed order.

    The caller owns the random generator, making augmentation reproducible from
    a recorded run seed, epoch, and sample index.
    """
    if recipe.per_box_rotation_degrees or recipe.per_box_translation_std:
        points, boxes = per_box_perturb(points, boxes, rng, recipe.per_box_rotation_degrees, recipe.per_box_translation_std)
    if rng.random() < recipe.flip_x_probability:
        points, boxes = flip_x(points, boxes)
    if rng.random() < recipe.flip_y_probability:
        points, boxes = flip_y(points, boxes)
    scale = rng.uniform(*recipe.scale_range)
    points, boxes = global_scale(points, boxes, scale)
    translation = rng.normal(0.0, np.asarray(recipe.translation_std), size=3)
    points, boxes = global_translate(points, boxes, translation)
    angle = np.deg2rad(rng.uniform(-recipe.global_yaw_degrees, recipe.global_yaw_degrees))
    return global_yaw_rotate(points, boxes, angle)
