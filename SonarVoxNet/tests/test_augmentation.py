import numpy as np

from sonarvoxnet.data.augmentation import AugmentationRecipe, Box3D, augment_sample, flip_x, flip_y, global_yaw_rotate, per_box_perturb
from sonarvoxnet.data.object_database import ObjectDatabaseEntry, insert_object
from sonarvoxnet.training.seed import augmentation_rng


def _box() -> Box3D:
    return Box3D(center=np.array([4.0, 1.0, 0.0]), dimensions=np.array([2.0, 1.0, 1.0]), rotation=np.eye(3))


def test_augmentation_seed_is_independent_of_worker_order():
    recipe = AugmentationRecipe(per_box_rotation_degrees=18.0, per_box_translation_std=0.2, flip_y_probability=0.5, scale_range=(0.95, 1.05))
    points = np.array([[4.0, 1.0, 0.0, 0.5], [4.3, 1.0, 0.0, 0.3]])
    first = augment_sample(points, [_box()], recipe, augmentation_rng(7, 3, 11))
    second = augment_sample(points, [_box()], recipe, augmentation_rng(7, 3, 11))
    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[1][0].center, second[1][0].center)
    np.testing.assert_allclose(first[1][0].rotation, second[1][0].rotation)


def test_y_flip_keeps_rotation_proper():
    _, boxes = flip_y(np.array([[4.0, 1.0, 0.0]]), [_box()])
    rotation = boxes[0].rotation
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-7)
    assert boxes[0].center[1] == -1.0


def test_candidate_global_yaw_and_front_back_flip_keep_rotation_proper():
    _, boxes = flip_x(np.array([[4.0, 1.0, 0.0]]), [_box()])
    _, boxes = global_yaw_rotate(np.array([[4.0, 1.0, 0.0]]), boxes, np.pi / 4)
    rotation = boxes[0].rotation
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-7)


def test_per_box_transform_moves_its_points_with_the_box():
    points = np.array([[4.0, 1.0, 0.0], [4.5, 1.0, 0.0]])
    augmented_points, augmented_boxes = per_box_perturb(points, [_box()], np.random.default_rng(4), 0.0, 0.5)
    displacement = augmented_boxes[0].center - _box().center
    np.testing.assert_allclose(augmented_points[:, :3] - points[:, :3], np.broadcast_to(displacement, (2, 3)))


def test_gt_sampling_rejects_a_colliding_candidate():
    points = np.array([[4.0, 1.0, 0.0]])
    entry = ObjectDatabaseEntry(local_points=np.array([[0.0, 0.0, 0.0]]), dimensions=np.array([2.0, 1.0, 1.0]), label="diver")
    boxes = [_box()]
    inserted_points, inserted_boxes = insert_object(points, boxes, entry, _box())
    np.testing.assert_array_equal(inserted_points, points)
    assert inserted_boxes is boxes
