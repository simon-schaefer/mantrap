import os
from typing import List

import numpy as np
import pytest

from mantrap.constants import sim_social_forces_default_params
from mantrap.simulation import SocialForcesSimulation
from mantrap.utility.io import path_from_home_directory
from mantrap.utility.shaping import check_ado_trajectories
from mantrap.utility.stats import Distribution, DirecDelta
from mantrap.evaluation.visualization import picture_opus


@pytest.mark.parametrize("goal_position", [np.array([2, 2]), np.array([0, -2])])
def test_single_ado_prediction(goal_position: np.ndarray):
    sim = SocialForcesSimulation()
    sim.add_ado(goal=goal_position, position=np.array([-1, -5]), velocity=np.ones(2) * 0.8, num_modes=1)

    trajectory = np.squeeze(sim.predict(t_horizon=100))
    assert np.isclose(trajectory[-1][0], goal_position[0], atol=0.5)
    assert np.isclose(trajectory[-1][1], goal_position[1], atol=0.5)


def test_static_ado_pair_prediction():
    sim = SocialForcesSimulation()
    sim.add_ado(goal=np.array([0, 0]), position=np.array([-1, 0]), velocity=np.array([0.1, 0]), num_modes=1)
    sim.add_ado(goal=np.array([0, 0]), position=np.array([1, 0]), velocity=np.array([-0.1, 0]), num_modes=1)

    trajectories = sim.predict(t_horizon=100)
    # Due to the repulsive of the agents between each other, they cannot both go to their goal position (which is
    # the same for both of them). Therefore the distance must be larger then zero basically, otherwise the repulsive
    # force would not act (or act attractive instead of repulsive).
    assert np.linalg.norm(trajectories[0, -1, 0:1] - trajectories[1, -1, 0:1]) > 1e-3


@pytest.mark.parametrize(
    "position, velocity, num_modes, v0s",
    [(np.array([-1, 0]), np.array([0.1, 0.2]), 2, [DirecDelta(2.3), DirecDelta(1.5)])],
)
def test_ado_ghosts_construction(position: np.ndarray, velocity: np.ndarray, num_modes: int, v0s: List[Distribution]):
    sim = SocialForcesSimulation()
    sim.add_ado(goal=np.zeros(2), position=position, velocity=velocity, num_modes=num_modes, v0s=v0s)

    assert sim.num_ado_modes == num_modes
    assert all([ghost.id == sim.ados[0].id for ghost in sim.ado_ghosts_agents])
    assert len(sim.ado_ghosts_agents) == num_modes

    assert all([type(v0) == DirecDelta for v0 in v0s])  # otherwise hard to compare due to sampling
    sim_v0s = [ghost.v0 for ghost in sim.ado_ghosts]
    sim_v0s_exp = [v0.mean for v0 in v0s]
    assert set(sim_v0s) == set(sim_v0s_exp)

    sim_sigmas = [ghost.sigma for ghost in sim.ado_ghosts]
    sim_sigmas_exp = [sim_social_forces_default_params["sigma"]] * num_modes
    assert set(sim_sigmas) == set(sim_sigmas_exp)


@pytest.mark.parametrize("num_modes, t_horizon, v0s", [(2, 4, [DirecDelta(2.3), DirecDelta(1.5)])])
def test_prediction_trajectories(num_modes: int, t_horizon: int, v0s: List[Distribution]):
    sim = SocialForcesSimulation()
    sim.add_ado(goal=np.ones(2), position=np.array([-1, 0]), velocity=np.array([1, 0]), num_modes=num_modes, v0s=v0s)
    sim.add_ado(goal=np.zeros(2), position=np.array([1, 0]), velocity=np.array([-1, 0]), num_modes=num_modes, v0s=v0s)

    ado_trajectories = sim.predict(t_horizon=t_horizon)
    assert check_ado_trajectories(ado_trajectories, t_horizon=t_horizon, num_modes=num_modes, num_ados=2)


@pytest.mark.parametrize("num_modes, t_horizon, v0s", [(2, 4, [DirecDelta(2.3), DirecDelta(1.5)])])
def test_prediction_one_agent_only(num_modes: int, t_horizon: int, v0s: List[Distribution]):
    sim = SocialForcesSimulation()
    sim.add_ado(goal=np.ones(2), position=np.array([-1, 0]), velocity=np.array([1, 0]), num_modes=num_modes, v0s=v0s)

    ado_trajectories = sim.predict(t_horizon=t_horizon)
    assert check_ado_trajectories(ado_trajectories, t_horizon=t_horizon, num_modes=num_modes, num_ados=1)
    assert np.array_equal(ado_trajectories[:, 0, :, :], ado_trajectories[:, 1, :, :])


def visualize_social_forces_multimodal():
    sim = SocialForcesSimulation(dt=0.5)
    v0s = [DirecDelta(0.1), DirecDelta(1.5)]
    sim.add_ado(goal=np.array([5, 0]), position=np.array([-5, 1]), velocity=np.array([1, 0]), num_modes=2, v0s=v0s)
    sim.add_ado(goal=np.array([-5, 0]), position=np.array([5, 0]), velocity=np.array([-1, 0]), num_modes=2, v0s=v0s)
    ado_trajectories = sim.predict(t_horizon=20)

    output_dir = path_from_home_directory(os.path.join("test", "graphs", "social_forces_multimodal"))
    picture_opus(output_dir, ado_trajectories, ado_colors=sim.ado_colors, ado_ids=sim.ado_ids)


if __name__ == '__main__':
    test_static_ado_pair_prediction()
