import torch

from sonarvoxnet.models.anchors import build_anchor_grid, decode_box_residuals, encode_box_residuals
from sonarvoxnet.models.heads import AnchorHead
from sonarvoxnet.models.model import SonarVoxNet
from sonarvoxnet.models.rotation import matrix_to_sixd
from sonarvoxnet.models.targets import project_to_zyaw, rotation_target
from sonarvoxnet.training.head_losses import anchor_head_loss


def test_anchor_grid_has_one_template_axis_and_metric_centers():
    anchors = build_anchor_grid(2, 3)
    assert anchors.shape == (2, 3, 2, 7)
    torch.testing.assert_close(anchors[0, 0, 0, :3], torch.tensor((0.1, -4.9, 0.0)))


def test_anchor_box_residual_round_trip_preserves_rotation_target():
    anchors = build_anchor_grid(1, 1).reshape(2, 7)
    boxes = torch.cat((anchors[:, :6] + torch.tensor((0.1, -0.2, 0.05, 0.1, 0.1, 0.1)), torch.randn(2, 6)), dim=-1)
    decoded = decode_box_residuals(anchors, encode_box_residuals(anchors, boxes))
    torch.testing.assert_close(decoded, boxes)


def test_anchor_head_and_loss_use_the_documented_shapes():
    head = AnchorHead(channels=8, num_anchors=2)
    prediction = head(torch.randn(3, 8, 4, 5))
    assert prediction["objectness"].shape == (3, 2, 4, 5)
    assert prediction["regression"].shape == (3, 2, 12, 4, 5)
    positive = torch.zeros(3, 2, 4, 5)
    positive[0, 0, 1, 2] = 1
    losses = anchor_head_loss(prediction, positive, torch.zeros_like(prediction["regression"]), positive)
    assert torch.isfinite(losses["total"])


def test_dense_middle_and_anchor_head_form_a_valid_ablation_path():
    model = SonarVoxNet(middle_encoder="dense", detection_head="anchor")
    features = torch.randn(8, 2, 7)
    num_points = torch.full((8,), 2)
    coordinates = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2], [0, 3, 3, 3],
         [0, 4, 4, 4], [0, 5, 5, 5], [0, 6, 6, 6], [0, 7, 7, 7]]
    )
    prediction = model(features, num_points, coordinates, spatial_shape=(8, 8, 8), batch_size=1)
    assert prediction["objectness"].shape[:2] == (1, 2)
    assert prediction["regression"].shape[2] == 12


def test_zyaw_projection_removes_pitch_and_roll():
    yaw = 0.4
    rotation = torch.tensor(
        [[torch.cos(torch.tensor(yaw)), -torch.sin(torch.tensor(yaw)), 0.0], [torch.sin(torch.tensor(yaw)), torch.cos(torch.tensor(yaw)), 0.0], [0.0, 0.0, 1.0]]
    )
    projected = project_to_zyaw(rotation)
    torch.testing.assert_close(rotation_target(rotation, "zyaw"), matrix_to_sixd(projected))
    torch.testing.assert_close(projected, rotation)
