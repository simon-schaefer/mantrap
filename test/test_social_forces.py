import time
from typing import List

import pytest
import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.constants import sim_social_forces_default_params
from mantrap.simulation import SocialForcesSimulation
from mantrap.utility.io import build_output_path
from mantrap.utility.primitives import straight_line_primitive
from mantrap.utility.shaping import check_trajectories
from mantrap.utility.maths import Distribution, DirecDelta


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


def test_build_graph_over_horizon():
    sim = SocialForcesSimulation(IntegratorDTAgent, {"position": torch.tensor([-5, 0])})
    sim.add_ado(position=torch.tensor([3, 0]), velocity=torch.zeros(2), goal=torch.tensor([-4, 0]), num_modes=2)
    sim.add_ado(position=torch.tensor([5, 0]), velocity=torch.zeros(2), goal=torch.tensor([-4, 0]), num_modes=2)
    sim.add_ado(position=torch.tensor([10, 0]), velocity=torch.zeros(2), goal=torch.tensor([-4, 0]), num_modes=2)

    prediction_horizon = 10
    ego_primitive = torch.ones((prediction_horizon, 2)) * sim.ego.position  # does not matter here anyway
    graphs = sim.build_connected_graph(ego_positions=ego_primitive)

    assert all([f"ego_{k}_position" in graphs.keys() for k in range(prediction_horizon)])
    assert all([f"ego_{k}_velocity" in graphs.keys() for k in range(prediction_horizon)])


@pytest.mark.parametrize("position, goal", [(torch.tensor([-5, 0]), torch.tensor([5, 0]))])
def test_ego_graph_updates(position: torch.Tensor, goal: torch.Tensor):
    sim = SocialForcesSimulation(IntegratorDTAgent, {"position": position, "velocity": torch.zeros(2)})
    primitives = straight_line_primitive(horizon=11, start_pos=position, end_pos=goal)

    graphs = sim.build_connected_graph(ego_positions=primitives)
    for k in range(primitives.shape[0]):
        assert torch.all(torch.eq(primitives[k, :], graphs[f"ego_{k}_position"]))


###########################################################################
# Visualizations ##########################################################
###########################################################################
def visualize_igrad_social_forces_computation_time():
    computational_times = {}
    t_horizons = range(1, 16, 2)
    nums_ados = [1, 3, 5, 10]

    for num_ados in nums_ados:
        computational_times[num_ados] = []
        for t_horizon in t_horizons:
            sim = SocialForcesSimulation(IntegratorDTAgent, {"position": torch.tensor([-5, 0])}, dt=0.2)
            for _ in range(num_ados):
                sim.add_ado(torch.rand(2) * 10, velocity=torch.rand(2), goal=torch.tensor([-4, 0]), num_modes=2)

            ego_primitive = torch.ones((t_horizon, 2)) * sim.ego.position  # does not matter here anyway
            start_time = time.time()
            sim.build_connected_graph(ego_positions=ego_primitive)
            computational_times[num_ados].append(time.time() - start_time)

    import matplotlib.pyplot as plt

    plt.Figure()
    plt.title("Social Forces Graph Building Time over full Prediction Horizon")
    for num_ados, times in computational_times.items():
        plt.plot(t_horizons, times, label=f"num_ados = {num_ados}")
    plt.xlabel("Prediction horizon [steps]")
    plt.ylabel("Runtime [s]")
    plt.legend()
    plt.savefig(build_output_path("test/graphs/igrad_social_forces_runtime.png", make_dir=False))
    plt.close()
