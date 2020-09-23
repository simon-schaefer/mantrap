import numpy as np
import pytest

import mantrap.agents
import mantrap.environment
import mantrap.utility.maths

from mantrap_evaluation.metrics import *

torch.manual_seed(0)


def expand_ado_trajectories(env: mantrap.environment.base.GraphBasedEnvironment, ado_trajectories: torch.Tensor
                            ) -> torch.Tensor:
    assert mantrap.utility.shaping.check_ado_trajectories(ado_trajectories)
    num_ados, num_samples, t_horizon, _ = ado_trajectories.shape
    trajectories_full = torch.zeros((num_ados, num_samples, t_horizon, 5))
    for m_ado, ado in enumerate(env.ados):
        for m_sample in range(num_samples):
            ado_path = ado_trajectories[m_ado, m_sample, :, 0:2]
            t_start = float(env.ados[m_ado].state_with_time[-1])
            t_horizon = ado_path.shape[0]
            trajectory = torch.zeros((t_horizon, 5))

            trajectory[:, 0:2] = ado_path
            trajectory[:-1, 2:4] = (trajectory[1:, 0:2] - trajectory[0:-1, 0:2]) / env.dt
            trajectory[:, 4] = torch.linspace(t_start, t_start + (t_horizon - 1) * env.dt, steps=t_horizon)

            trajectories_full[m_ado, m_sample, :, :] = trajectory
    return trajectories_full


def test_minimal_distance_principle():
    ego_trajectory = mantrap.utility.maths.straight_line(torch.tensor([-5, 0.1]), torch.tensor([5, 0.1]), steps=10)

    ado_trajectory_1 = torch.zeros((1, 10, 1, 2))
    distance = metric_minimal_distance(ego_trajectory=ego_trajectory, ado_trajectories=ado_trajectory_1)
    assert np.isclose(distance, 0.1, atol=1e-3)

    ado_trajectory_2 = mantrap.utility.maths.straight_line(torch.ones(2), torch.ones(2) * 10, steps=10)
    ado_trajectory_2 = ado_trajectory_2.view(1, -1, 1, 2)
    ado_trajectory_2 = torch.cat((ado_trajectory_1, ado_trajectory_2))
    distance = metric_minimal_distance(ego_trajectory=ego_trajectory, ado_trajectories=ado_trajectory_2)
    assert np.isclose(distance, 0.1, atol=1e-3)

    ado_trajectory_3 = mantrap.utility.maths.straight_line(torch.tensor([10, 8]),  torch.tensor([-10, 8]), steps=10)
    ado_trajectory_3[5, :] = torch.tensor([5, 0.1])
    ado_trajectory_3 = ado_trajectory_3.view(1, -1, 1, 2)
    distance = metric_minimal_distance(ego_trajectory=ego_trajectory, ado_trajectories=ado_trajectory_3)
    assert not np.isclose(distance, 0.0, atol=1e-3)  # tolerance, not time-equivalent at (5, 0.1) (!)

    ado_trajectory_3 = mantrap.utility.maths.straight_line(torch.tensor([10, 8]),  torch.tensor([-10, 8]), steps=10)
    ado_trajectory_3[-1, :] = torch.tensor([5, 0.1])
    ado_trajectory_3 = ado_trajectory_3.view(1, -1, 1, 2)
    distance = metric_minimal_distance(ego_trajectory=ego_trajectory, ado_trajectories=ado_trajectory_3)
    assert np.isclose(distance, 0.0, atol=1e-3)  # now time-equivalent at (5, 0.1) (!)


def test_minimal_distance_interpolation():
    ego_traj = mantrap.utility.maths.straight_line(torch.tensor([-5, 0.0]), torch.tensor([5, 0.0]), steps=10)
    ado_traj = mantrap.utility.maths.straight_line(torch.tensor([0.0, -5]), torch.tensor([0.0, 5]), steps=10)

    # Both trajectories dont pass through the origin in discrete space due to the discretization pattern, therefore
    # it's minimal distance in discrete time should be larger than zero.
    min_distance_dt = torch.min(torch.norm(ego_traj - ado_traj, dim=1)).item()
    assert not np.isclose(min_distance_dt, 0.0, atol=0.1)

    # However using the interpolation scheme as running in the metric, they should cross each other.
    ado_traj = ado_traj.view(1, -1, 1, 2)
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
                                       mantrap.environment.SocialForcesEnvironment])
def test_ado_effort(env_class: mantrap.environment.base.GraphBasedEnvironment.__class__):
    env = env_class(ego_type=mantrap.agents.DoubleIntegratorDTAgent, ego_position=torch.tensor([5, 0]))
    env.add_ado(position=torch.zeros(2), velocity=torch.tensor([1, 0]))

    # When the ado trajectories are exactly the same as predicting them without an ego, the score should be zero.
    ado_trajectories = env.predict_wo_ego(t_horizon=10)
    metric_score_wo = metric_ado_effort(ado_trajectories=ado_trajectories, env=env)

    # Otherwise it is very hard to predict the exact score, but we know it should be non-zero and positive.
    ado_trajectories = env.predict_w_controls(ego_controls=torch.ones((5, 2)))
    metric_score = metric_ado_effort(ado_trajectories=ado_trajectories, env=env)
    assert metric_score >= metric_score_wo * 0.5


def test_final_distance():
    ego_trajectory = torch.rand((10, 5))
    ego_trajectory[0, 0:2] = torch.tensor([0, 0])
    ego_trajectory[-1, 0:2] = torch.tensor([5, 0])
    goal = torch.tensor([10, 0])

    score = metric_final_distance(ego_trajectory, goal=goal)
    assert np.isclose(score, 0.5)


def test_extra_time():
    env = mantrap.environment.PotentialFieldEnvironment(ego_position=torch.zeros(2))
    goal = torch.tensor([5, 0])
    solver = mantrap.solver.IPOPTSolver(env=env, goal=goal, modules=[mantrap.modules.GoalNormModule,
                                                                     mantrap.modules.SpeedLimitModule])

    # The first tested trajectory goes directly to the goal as fast as possible.
    ego_trajectory_1, _ = solver.solve(time_steps=5)
    score = metric_extra_time(ego_trajectory=ego_trajectory_1, env=env)
    assert np.isclose(score, 0.0)
