"""Dataset-independent geometric augmentation utilities."""

from sonarvoxnet.data.augmentation import AugmentationRecipe, Box3D, augment_sample
from sonarvoxnet.data.object_database import ObjectDatabaseEntry, build_object_database, insert_object

__all__ = ["AugmentationRecipe", "Box3D", "ObjectDatabaseEntry", "augment_sample", "build_object_database", "insert_object"]
