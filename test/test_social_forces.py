import os
from typing import List

import pytest
import torch

from mantrap.constants import sim_social_forces_default_params
from mantrap.simulation import SocialForcesSimulation
from mantrap.utility.io import path_from_home_directory
from mantrap.utility.shaping import check_trajectories
from mantrap.utility.stats import Distribution, DirecDelta


@pytest.mark.parametrize("goal_position", [torch.tensor([2, 2]), torch.tensor([0, -2])])
def test_single_ado_prediction(goal_position: torch.Tensor):
    sim = SocialForcesSimulation()
    sim.add_ado(goal=goal_position, position=torch.tensor([-1, -5]), velocity=torch.ones(2) * 0.8, num_modes=1)

    trajectory = torch.squeeze(sim.predict(t_horizon=100))
    assert torch.isclose(trajectory[-1][0], goal_position[0].float(), atol=0.5)
    assert torch.isclose(trajectory[-1][1], goal_position[1].float(), atol=0.5)


def test_static_ado_pair_prediction():
    sim = SocialForcesSimulation()
    sim.add_ado(goal=torch.zeros(2), position=torch.tensor([-1, 0]), velocity=torch.tensor([0.1, 0]), num_modes=1)
    sim.add_ado(goal=torch.zeros(2), position=torch.tensor([1, 0]), velocity=torch.tensor([-0.1, 0]), num_modes=1)

    trajectories = sim.predict(t_horizon=100)
    # Due to the repulsive of the agents between each other, they cannot both go to their goal position (which is
    # the same for both of them). Therefore the distance must be larger then zero basically, otherwise the repulsive
    # force would not act (or act attractive instead of repulsive).
    assert torch.norm(trajectories[0, -1, 0:1] - trajectories[1, -1, 0:1]) > 1e-3


@pytest.mark.parametrize(
    "pos, vel, num_modes, v0s",
    [(torch.tensor([-1, 0]), torch.tensor([0.1, 0.2]), 2, [DirecDelta(2.3), DirecDelta(1.5)])],
)
def test_ado_ghosts_construction(pos: torch.Tensor, vel: torch.Tensor, num_modes: int, v0s: List[Distribution]):
    sim = SocialForcesSimulation()
    sim.add_ado(goal=torch.zeros(2), position=pos, velocity=vel, num_modes=num_modes, v0s=v0s)

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
def test_prediction_trajectories_shape(num_modes: int, t_horizon: int, v0s: List[Distribution]):
    sim = SocialForcesSimulation()
    sim.add_ado(goal=torch.ones(2), position=torch.tensor([-1, 0]), num_modes=num_modes, v0s=v0s)
    sim.add_ado(goal=torch.zeros(2), position=torch.tensor([1, 0]), num_modes=num_modes, v0s=v0s)

    ado_trajectories = sim.predict(t_horizon=t_horizon)
    assert check_trajectories(ado_trajectories, t_horizon=t_horizon, modes=num_modes, ados=2)


@pytest.mark.parametrize("num_modes, t_horizon, v0s", [(2, 4, [DirecDelta(2.3), DirecDelta(1.5)])])
def test_prediction_one_agent_only(num_modes: int, t_horizon: int, v0s: List[Distribution]):
    sim = SocialForcesSimulation()
    sim.add_ado(goal=torch.ones(2), position=torch.ones(2), velocity=torch.tensor([1, 0]), num_modes=num_modes, v0s=v0s)

    ado_trajectories = sim.predict(t_horizon=t_horizon)
    assert check_trajectories(ado_trajectories, t_horizon=t_horizon, modes=num_modes, ados=1)
    assert torch.all(torch.eq(ado_trajectories[:, 0, :, :], ado_trajectories[:, 1, :, :]))


###########################################################################
# Visualizations ##########################################################
###########################################################################


def visualize_social_forces_multimodal():
    sim = SocialForcesSimulation(dt=0.5)
    v0s = [DirecDelta(0.1), DirecDelta(1.5)]

    velocity = torch.tensor([1, 0])
    sim.add_ado(goal=torch.tensor([5, 0]), position=torch.tensor([-5, 1]), velocity=velocity, num_modes=2, v0s=v0s)
    sim.add_ado(goal=torch.tensor([-5, 0]), position=torch.tensor([5, 0]), velocity=velocity * -1, num_modes=2, v0s=v0s)
    ado_trajectories = sim.predict(t_horizon=20)

    from mantrap.evaluation.visualization import picture_opus
    output_dir = path_from_home_directory(os.path.join("test", "graphs", "social_forces_multimodal"))
    picture_opus(output_dir, ado_trajectories, ado_colors=sim.ado_colors, ado_ids=sim.ado_ids)
