import torch

from sonarvoxnet.evaluation.metrics import geodesic_rotation_error_degrees


def test_identical_rotations_have_zero_geodesic_error():
    identity = torch.eye(3).unsqueeze(0)
    torch.testing.assert_close(geodesic_rotation_error_degrees(identity, identity), torch.zeros(1), atol=1e-5, rtol=0)
