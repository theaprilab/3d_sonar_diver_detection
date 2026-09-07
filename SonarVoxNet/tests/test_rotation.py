import torch

from sonarvoxnet.models.rotation import matrix_to_sixd, sixd_to_matrix


def test_sixd_identity_round_trip():
    identity = torch.eye(3).unsqueeze(0)
    decoded = sixd_to_matrix(matrix_to_sixd(identity))
    torch.testing.assert_close(decoded, identity, atol=1e-6, rtol=1e-6)


def test_sixd_decoder_produces_proper_rotation():
    matrix = sixd_to_matrix(torch.randn(8, 6))
    torch.testing.assert_close(matrix.transpose(-1, -2) @ matrix, torch.eye(3).expand(8, 3, 3), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.det(matrix), torch.ones(8), atol=1e-5, rtol=1e-5)
