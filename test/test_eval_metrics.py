import numpy as np
import torch

from mantrap.evaluation.metrics import metric_ego_effort, metric_minimal_distance
from mantrap.utility.primitives import straight_line
from mantrap.utility.utility import build_trajectory_from_path


def test_minimal_distance_principle():
    ego_trajectory = straight_line(torch.tensor([-5, 0.1]), torch.tensor([5, 0.1]), steps=10)

    ado_traj_1 = torch.zeros((1, 1, 10, 2))
    distance = metric_minimal_distance(ego_trajectory=ego_trajectory, ado_trajectories=ado_traj_1)
    assert np.isclose(distance, 0.1, atol=1e-3)

    ado_traj_2 = torch.cat((ado_traj_1, straight_line(torch.ones(2), torch.ones(2) * 10, steps=10).view(1, 1, -1, 2)))
    distance = metric_minimal_distance(ego_trajectory=ego_trajectory, ado_trajectories=ado_traj_2)
    assert np.isclose(distance, 0.1, atol=1e-3)

    ado_traj_3 = straight_line(torch.tensor([10, 8]),  torch.tensor([-10, 8]), steps=10)
    ado_traj_3[5, :] = torch.tensor([5, 0.1])
    distance = metric_minimal_distance(ego_trajectory=ego_trajectory, ado_trajectories=ado_traj_3.view(1, 1, -1, 2))
    assert not np.isclose(distance, 0.0, atol=1e-3)  # tolerance, not time-equivalent at (5, 0.1) (!)

    ado_traj_3 = straight_line(torch.tensor([10, 8]),  torch.tensor([-10, 8]), steps=10)
    ado_traj_3[-1, :] = torch.tensor([5, 0.1])
    distance = metric_minimal_distance(ego_trajectory=ego_trajectory, ado_trajectories=ado_traj_3.view(1, 1, -1, 2))
    assert np.isclose(distance, 0.0, atol=1e-3)  # now time-equivalent at (5, 0.1) (!)


def test_minimal_distance_interpolation():
    ego_traj = straight_line(torch.tensor([-5, 0.0]), torch.tensor([5, 0.0]), steps=10)
    ado_traj = straight_line(torch.tensor([0.0, -5]), torch.tensor([0.0, 5]), steps=10)

    # Both trajectories dont pass through the origin in discrete space due to the discretization pattern, therefore
    # it's minimal distance in discrete time should be larger than zero.
    min_distance_dt = torch.min(torch.norm(ego_traj - ado_traj, dim=1)).item()
    assert not np.isclose(min_distance_dt, 0.0, atol=0.1)

    # However using the interpolation scheme as running in the metric, they should cross each other.
    ado_traj = ado_traj.view(1, 1, -1, 2)
    min_distance_ct = metric_minimal_distance(ego_trajectory=ego_traj, ado_trajectories=ado_traj, num_inter_points=1000)
    assert np.isclose(min_distance_ct, 0.0, atol=1e-3)


def test_ego_effort():
    ego_trajectory = straight_line(torch.tensor([-5, 0.1]), torch.tensor([5, 0.1]), steps=10)
    ego_trajectory = build_trajectory_from_path(ego_trajectory, dt=1.0)
    effort = metric_ego_effort(ego_trajectory=ego_trajectory)
    assert np.isclose(effort, 0.0, atol=1e-3)
