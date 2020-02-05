import torch


def test_view_and_review():
    x = torch.rand((20, 2))
    x_flat = x.flatten()
    assert torch.all(torch.eq(x, x_flat.view(-1, 2)))
