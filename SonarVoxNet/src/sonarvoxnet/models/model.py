"""Core VFE, sparse middle encoder, BEV backbone, and detection head."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional

from sonarvoxnet.models.heads import AnchorHead
from sonarvoxnet.models.rotation import CHANNELS


class VFELayer(nn.Module):
    """Point-wise FCN followed by voxel-wise max pooling and feature concatenation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if out_channels % 2:
            raise ValueError("VFE output channels must be even.")
        self.linear = nn.Linear(in_channels, out_channels // 2)
        self.normalization = nn.BatchNorm1d(out_channels // 2)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        voxels, points, _ = features.shape
        flat_mask = mask.reshape(voxels * points)
        pointwise = features.new_zeros(voxels * points, self.linear.out_features)
        if flat_mask.any():
            valid = features.reshape(voxels * points, -1)[flat_mask]
            pointwise[flat_mask] = functional.relu(self.normalization(self.linear(valid)))
        pointwise = pointwise.reshape(voxels, points, -1)
        pooled = pointwise.masked_fill(~mask.unsqueeze(-1), torch.finfo(pointwise.dtype).min).max(1, keepdim=True).values.clamp_min(0)
        return torch.cat((pointwise, pooled.expand(-1, points, -1)), dim=-1) * mask.unsqueeze(-1)


class VFE(nn.Module):
    """Two VFE layers and a final voxel-wise feature projection."""

    def __init__(self, input_channels: int = 7, output_channels: int = 128) -> None:
        super().__init__()
        self.first = VFELayer(input_channels, 32)
        self.second = VFELayer(32, 128)
        self.linear = nn.Linear(128, output_channels)
        self.normalization = nn.BatchNorm1d(output_channels)

    def forward(self, features: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        points_per_voxel = features.shape[1]
        mask = torch.arange(points_per_voxel, device=features.device)[None] < num_points[:, None]
        features = self.second(self.first(features, mask), mask)
        valid = mask.reshape(-1)
        projected = features.new_zeros(features.shape[0] * points_per_voxel, self.linear.out_features)
        if valid.any():
            projected[valid] = functional.relu(self.normalization(self.linear(features.reshape(-1, features.shape[-1])[valid])))
        projected = projected.reshape(features.shape[0], points_per_voxel, -1)
        return projected.masked_fill(~mask.unsqueeze(-1), torch.finfo(projected.dtype).min).max(1).values.clamp_min(0)


class DenseMiddleEncoder(nn.Module):
    """Dense Conv3D middle encoder retained for the dense-versus-sparse ablation."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(128, 64, 3, stride=(2, 1, 1), padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding=(0, 1, 1)), nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 64, 3, stride=(2, 1, 1), padding=1), nn.BatchNorm3d(64), nn.ReLU(),
        )

    def forward(self, dense_voxels: torch.Tensor) -> torch.Tensor:
        return self.layers(dense_voxels)


class SparseMiddleEncoder(nn.Module):
    """Sparse 3D middle encoder with the same dense output shape as the ablation encoder."""

    def __init__(self) -> None:
        super().__init__()
        try:
            import spconv.pytorch as spconv
        except ImportError as error:
            raise ImportError("The proposed model requires a compatible spconv installation. Install the project's sparse extra.") from error
        self.spconv = spconv
        self.layers = spconv.SparseSequential(
            spconv.SubMConv3d(128, 64, 3, padding=1, bias=False, indice_key="subm_1"), nn.BatchNorm1d(64), nn.ReLU(),
            spconv.SubMConv3d(64, 64, 3, padding=1, bias=False, indice_key="subm_1"), nn.BatchNorm1d(64), nn.ReLU(),
            spconv.SparseConv3d(64, 64, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=False), nn.BatchNorm1d(64), nn.ReLU(),
            spconv.SubMConv3d(64, 64, 3, padding=1, bias=False, indice_key="subm_2"), nn.BatchNorm1d(64), nn.ReLU(),
            spconv.SparseConv3d(64, 64, (3, 1, 1), padding=0, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
            spconv.SparseConv3d(64, 64, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=False), nn.BatchNorm1d(64), nn.ReLU(),
        )

    def forward(self, features: torch.Tensor, coordinates: torch.Tensor, spatial_shape: tuple[int, int, int], batch_size: int) -> torch.Tensor:
        tensor = self.spconv.SparseConvTensor(features, coordinates.int(), spatial_shape, batch_size)
        return self.layers(tensor).dense()


class BEVBackbone(nn.Module):
    """BEV convolutional block shared by the proposed and ablation models.

    The first convolution is lazy because the collapsed vertical dimension is
    determined by the voxel grid and middle encoder, rather than by the head.
    """

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LazyConv2d(channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.block(features)


class CenterHead(nn.Module):
    """Anchor-free center head with full-3D rotation regression."""

    def __init__(self, channels: int = 128, rotation_representation: str = "6d") -> None:
        super().__init__()
        if rotation_representation not in CHANNELS:
            raise ValueError(f"Unsupported rotation representation: {rotation_representation}")
        self.heatmap = nn.Conv2d(channels, 1, 1)
        self.offset = nn.Conv2d(channels, 2, 1)
        self.height = nn.Conv2d(channels, 1, 1)
        self.dimensions = nn.Conv2d(channels, 3, 1)
        self.rotation = nn.Conv2d(channels, CHANNELS[rotation_representation], 1)
        nn.init.constant_(self.heatmap.bias, -math.log(9.0))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"heatmap": self.heatmap(features), "offset": self.offset(features), "height": self.height(features), "dimensions": self.dimensions(features), "rotation": self.rotation(features)}


class SonarVoxNet(nn.Module):
    """The paper model; sparse is the default and dense is an explicit ablation."""

    def __init__(
        self,
        middle_encoder: str = "sparse",
        detection_head: str = "center",
        rotation_representation: str = "6d",
        num_anchors: int = 2,
    ) -> None:
        super().__init__()
        if middle_encoder not in {"sparse", "dense"}:
            raise ValueError("middle_encoder must be 'sparse' or 'dense'.")
        if detection_head not in {"center", "anchor"}:
            raise ValueError("detection_head must be 'center' or 'anchor'.")
        self.vfe = VFE()
        self.middle_encoder_name = middle_encoder
        self.middle = SparseMiddleEncoder() if middle_encoder == "sparse" else DenseMiddleEncoder()
        self.bev_backbone = BEVBackbone()
        self.head_name = detection_head
        self.head = (
            CenterHead(rotation_representation=rotation_representation)
            if detection_head == "center"
            else AnchorHead(num_anchors=num_anchors, rotation_representation=rotation_representation)
        )

    def forward(self, voxel_features: torch.Tensor, num_points: torch.Tensor, coordinates: torch.Tensor, spatial_shape: tuple[int, int, int], batch_size: int) -> dict[str, torch.Tensor]:
        voxel_features = self.vfe(voxel_features, num_points)
        if self.middle_encoder_name == "sparse":
            middle = self.middle(voxel_features, coordinates, spatial_shape, batch_size)
        else:
            dense = voxel_features.new_zeros(batch_size, 128, *spatial_shape)
            if coordinates.numel():
                batch, depth, row, column = coordinates.unbind(dim=1)
                dense[batch, :, depth, row, column] = voxel_features
            middle = self.middle(dense)
        batch, channels, depth, height, width = middle.shape
        return self.head(self.bev_backbone(middle.reshape(batch, channels * depth, height, width)))
