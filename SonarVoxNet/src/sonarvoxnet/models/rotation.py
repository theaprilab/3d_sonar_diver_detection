"""Continuous representations and decoders for rotations in SO(3)."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


CHANNELS = {"6d": 6, "quaternion": 4, "axis_angle": 3, "9d": 9}


def sixd_to_matrix(value: torch.Tensor) -> torch.Tensor:
    """Convert two unconstrained 3-vectors to a rotation matrix by Gram--Schmidt."""
    first, second = value[..., :3], value[..., 3:]
    basis_1 = functional.normalize(first, dim=-1, eps=1e-8)
    second = second - (basis_1 * second).sum(dim=-1, keepdim=True) * basis_1
    basis_2 = functional.normalize(second, dim=-1, eps=1e-8)
    basis_3 = torch.cross(basis_1, basis_2, dim=-1)
    return torch.stack((basis_1, basis_2, basis_3), dim=-1)


def matrix_to_sixd(matrix: torch.Tensor) -> torch.Tensor:
    """Store the first two columns of a rotation matrix."""
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def quaternion_to_matrix(value: torch.Tensor) -> torch.Tensor:
    """Decode a scalar-first quaternion after normalizing its magnitude."""
    w, x, y, z = functional.normalize(value, dim=-1, eps=1e-8).unbind(dim=-1)
    row_1 = torch.stack((1 - 2 * (y.square() + z.square()), 2 * (x * y - z * w), 2 * (x * z + y * w)), dim=-1)
    row_2 = torch.stack((2 * (x * y + z * w), 1 - 2 * (x.square() + z.square()), 2 * (y * z - x * w)), dim=-1)
    row_3 = torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x.square() + y.square())), dim=-1)
    return torch.stack((row_1, row_2, row_3), dim=-2)


def axis_angle_to_matrix(value: torch.Tensor) -> torch.Tensor:
    """Decode an axis-angle vector using Rodrigues' formula."""
    angle = torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = value / angle
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((torch.stack((zero, -z, y), -1), torch.stack((z, zero, -x), -1), torch.stack((-y, x, zero), -1)), -2)
    identity = torch.eye(3, dtype=value.dtype, device=value.device).expand(*value.shape[:-1], 3, 3)
    sine = torch.sin(angle)[..., None]
    cosine = torch.cos(angle)[..., None]
    return identity + sine * skew + (1 - cosine) * (skew @ skew)


def nined_to_matrix(value: torch.Tensor) -> torch.Tensor:
    """Project an unconstrained 3-by-3 matrix to the closest proper rotation."""
    matrix = value.reshape(*value.shape[:-1], 3, 3)
    left, _, right = torch.linalg.svd(matrix)
    rotation = left @ right
    sign = torch.where(torch.det(rotation) < 0, -torch.ones_like(torch.det(rotation)), torch.ones_like(torch.det(rotation)))
    left = left.clone()
    left[..., :, -1] *= sign.unsqueeze(-1)
    return left @ right


def decode(value: torch.Tensor, representation: str) -> torch.Tensor:
    """Decode one of the registered rotation representations to a matrix."""
    decoders = {"6d": sixd_to_matrix, "quaternion": quaternion_to_matrix, "axis_angle": axis_angle_to_matrix, "9d": nined_to_matrix}
    try:
        return decoders[representation](value)
    except KeyError as error:
        raise ValueError(f"Unsupported rotation representation: {representation}") from error
