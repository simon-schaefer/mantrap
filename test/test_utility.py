import torch

from mantrap.utility.primitives import straight_line_primitive
from mantrap.utility.utility import build_trajectory_from_positions


def test_build_trajectory_from_positions():
    positions = straight_line_primitive(prediction_horizon=11, start_pos=torch.zeros(2), end_pos=torch.ones(2) * 10)
    trajectory = build_trajectory_from_positions(positions, dt=1.0, t_start=0.0)

    assert torch.all(torch.eq(trajectory[:, 0:2], positions))
    assert torch.all(torch.eq(trajectory[1:, 3:5], torch.ones(10, 2)))
