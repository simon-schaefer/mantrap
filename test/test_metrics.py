import numpy as np
import pytest

import mantrap.agents
import mantrap.environment
import mantrap.utility.maths

from mantrap_evaluation.metrics import *


def test_minimal_distance_principle():
    ego_trajectory = mantrap.utility.maths.straight_line(torch.tensor([-5, 0.1]), torch.tensor([5, 0.1]), steps=10)

    ado_traj_1 = torch.zeros((1, 1, 10, 2))
    distance = metric_minimal_distance(ego_trajectory=ego_trajectory, ado_trajectories=ado_traj_1)
    assert np.isclose(distance, 0.1, atol=1e-3)

    ado_traj_2 = mantrap.utility.maths.straight_line(torch.ones(2), torch.ones(2) * 10, steps=10).view(1, 1, -1, 2)
    ado_traj_2 = torch.cat((ado_traj_1, ado_traj_2))
    distance = metric_minimal_distance(ego_trajectory=ego_trajectory, ado_trajectories=ado_traj_2)
    assert np.isclose(distance, 0.1, atol=1e-3)

    ado_traj_3 = mantrap.utility.maths.straight_line(torch.tensor([10, 8]),  torch.tensor([-10, 8]), steps=10)
    ado_traj_3[5, :] = torch.tensor([5, 0.1])
    distance = metric_minimal_distance(ego_trajectory=ego_trajectory, ado_trajectories=ado_traj_3.view(1, 1, -1, 2))
    assert not np.isclose(distance, 0.0, atol=1e-3)  # tolerance, not time-equivalent at (5, 0.1) (!)

    ado_traj_3 = mantrap.utility.maths.straight_line(torch.tensor([10, 8]),  torch.tensor([-10, 8]), steps=10)
    ado_traj_3[-1, :] = torch.tensor([5, 0.1])
    distance = metric_minimal_distance(ego_trajectory=ego_trajectory, ado_trajectories=ado_traj_3.view(1, 1, -1, 2))
    assert np.isclose(distance, 0.0, atol=1e-3)  # now time-equivalent at (5, 0.1) (!)


def test_minimal_distance_interpolation():
    ego_traj = mantrap.utility.maths.straight_line(torch.tensor([-5, 0.0]), torch.tensor([5, 0.0]), steps=10)
    ado_traj = mantrap.utility.maths.straight_line(torch.tensor([0.0, -5]), torch.tensor([0.0, 5]), steps=10)

    # Both trajectories dont pass through the origin in discrete space due to the discretization pattern, therefore
    # it's minimal distance in discrete time should be larger than zero.
    min_distance_dt = torch.min(torch.norm(ego_traj - ado_traj, dim=1)).item()
    assert not np.isclose(min_distance_dt, 0.0, atol=0.1)

    # However using the interpolation scheme as running in the metric, they should cross each other.
    ado_traj = ado_traj.view(1, 1, -1, 2)
    min_distance_ct = metric_minimal_distance(ego_trajectory=ego_traj, ado_trajectories=ado_traj, num_inter_points=1000)
    assert np.isclose(min_distance_ct, 0.0, atol=1e-3)


@pytest.mark.parametrize(
    "controls, effort_score",
    [
        (torch.stack((torch.ones(5) * 2.0, torch.zeros(5)), dim=1), 1.0),
        (torch.zeros((5, 2)), 0.0),
        (torch.stack((torch.tensor([2.0, 2.0, 1.0, 2.0]), torch.zeros(4)), dim=1), 0.875)  # 7/8
    ]
)
def test_ego_effort(controls: torch.Tensor, effort_score: float):
    ego = mantrap.agents.DoubleIntegratorDTAgent(position=torch.zeros(2))
    ego_trajectory = ego.unroll_trajectory(controls=controls, dt=1.0)

    metric_score = metric_ego_effort(ego_trajectory=ego_trajectory, max_acceleration=2.0)
    assert np.isclose(metric_score, effort_score)


@pytest.mark.parametrize(
    "velocity_profiles, directness_score",
    [
        (torch.stack((torch.ones(5) * 2.0, torch.zeros(5)), dim=1), 1.0),
        (torch.stack((torch.ones(5) * 1.49, torch.zeros(5)), dim=1), 1.0),
        (torch.stack((torch.ones(5) * (-2.0), torch.zeros(5)), dim=1), -1.0),
        (torch.zeros((5, 2)), 0.0),
        (torch.ones((5, 2)) * 2, np.cos(np.pi / 4))  # length of arrow in x direction of (0, 0) -> (1, 1) vector
    ]
)
def test_directness(velocity_profiles: torch.Tensor, directness_score: float):
    start, goal = torch.zeros(2), torch.tensor([10.0, 0.0])
    ego_trajectory = torch.zeros((velocity_profiles.shape[0], 5))
    ego_trajectory[:, 2:4] = velocity_profiles  # the remaining data is not used anyways (just checked for shape sanity)

    metric_score = metric_directness(ego_trajectory=ego_trajectory, goal=goal)
    assert np.isclose(metric_score, directness_score)


@pytest.mark.parametrize("env_class", [mantrap.environment.KalmanEnvironment,
                                       mantrap.environment.PotentialFieldEnvironment,
                                       mantrap.environment.SocialForcesEnvironment,
                                       mantrap.environment.ORCAEnvironment,
                                       mantrap.environment.Trajectron])
def test_ado_effort(env_class: mantrap.environment.base.GraphBasedEnvironment.__class__):
    env = env_class(mantrap.agents.DoubleIntegratorDTAgent, {"position": torch.tensor([5, 0])})
    env.add_ado(position=torch.zeros(2), velocity=torch.tensor([1, 0]))

    # When the ado trajectories are exactly the same as predicting them without an ego, the score should be zero.
    ado_traj = env.predict_wo_ego(t_horizon=10)
    metric_score = metric_ado_effort(ado_trajectories=ado_traj, env=env)
    if env.is_deterministic:
        assert np.isclose(metric_score, 0.0)

    # Otherwise it is very hard to predict the exact score, but we know it should be non-zero and positive.
    ado_traj = env.predict_w_controls(ego_controls=torch.ones(5, 2))
    metric_score = metric_ado_effort(ado_trajectories=ado_traj, env=env)
    assert metric_score >= 0.0

    # For testing the effect of re-predicting the ado trajectories without an ego for the current scene state we
    # stack an altered ado trajectory tensor to another one with is exactly the same as without ego in the scene.
    # When the environment is correctly reset at every time-step, then the only contribution to the ado effort comes
    # from the first part of the ado trajectory (which was predicted with an ego agent in the scene). Therefore the
    # score w.r.t. only the first part and for the combined ado trajectory should be the same.
    env_test = env.copy()

    ado_traj_1 = env.predict_w_controls(ego_controls=torch.ones(3, 2)).detach()
    metric_score_1 = metric_ado_effort(ado_trajectories=ado_traj_1, env=env)

    env_test.step_reset(ego_state_next=None, ado_states_next=ado_traj_1[:, 0, -1, :])
    ado_traj_2 = env_test.predict_wo_ego(t_horizon=4).detach()
    ado_traj_12 = torch.cat((ado_traj_1, ado_traj_2), dim=2)
    ado_traj_12[:, :, :, -1] = torch.linspace(0, 7 * env.dt, steps=8)
    metric_score_12 = metric_ado_effort(ado_trajectories=ado_traj_12, env=env)
    if env.is_deterministic:
        assert np.isclose(metric_score_1, metric_score_12, atol=1e-3)
