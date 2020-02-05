import time

import pytest
import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.simulation import SocialForcesSimulation
from mantrap.solver import IGradSolver
from mantrap.utility.io import path_from_home_directory
from mantrap.utility.primitives import straight_line_primitive


def test_build_graph_over_horizon():
    sim = SocialForcesSimulation(IntegratorDTAgent, {"position": torch.tensor([-5, 0])})
    sim.add_ado(position=torch.tensor([3, 0]), velocity=torch.zeros(2), goal=torch.tensor([-4, 0]), num_modes=2)
    sim.add_ado(position=torch.tensor([5, 0]), velocity=torch.zeros(2), goal=torch.tensor([-4, 0]), num_modes=2)
    sim.add_ado(position=torch.tensor([10, 0]), velocity=torch.zeros(2), goal=torch.tensor([-4, 0]), num_modes=2)

    prediction_horizon = 10
    ego_primitive = torch.ones((prediction_horizon, 2)) * sim.ego.position  # does not matter here anyway
    solver = IGradSolver(sim, goal=torch.zeros(2))

    assert torch.all(torch.eq(sim.ego.position, solver.env.ego.position))
    assert torch.all(torch.eq(sim.ego.velocity, solver.env.ego.velocity))

    graphs = solver.build_connected_graph(sim, ego_primitive)

    assert len(graphs) == prediction_horizon
    assert all(["ego_position" in graph.keys() for graph in graphs])
    assert all(["ego_velocity" in graph.keys() for graph in graphs])


@pytest.mark.parametrize("position, goal", [(torch.tensor([-5, 0]), torch.tensor([5, 0]))])
def test_ego_graph_updates(position: torch.Tensor, goal: torch.Tensor):
    sim = SocialForcesSimulation(IntegratorDTAgent, {"position": position, "velocity": torch.zeros(2)})
    primitives = straight_line_primitive(prediction_horizon=11, start_pos=position, end_pos=goal)

    solver = IGradSolver(sim, goal=goal)
    graphs = solver.build_connected_graph(sim, ego_positions=primitives)
    for i, graph in enumerate(graphs):
        assert torch.all(torch.eq(primitives[i, :], graph["ego_position"]))


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
            solver = IGradSolver(sim, goal=torch.zeros(2))

            ego_primitive = torch.ones((t_horizon, 2)) * sim.ego.position  # does not matter here anyway
            start_time = time.time()
            solver.build_connected_graph(sim, ego_primitive)
            computational_times[num_ados].append(time.time() - start_time)

    import matplotlib.pyplot as plt

    plt.Figure()
    plt.title("Social Forces Graph Building Time over full Prediction Horizon")
    for num_ados, times in computational_times.items():
        plt.plot(t_horizons, times, label=f"num_ados = {num_ados}")
    plt.xlabel("Prediction horizon [steps]")
    plt.ylabel("Runtime [s]")
    plt.legend()
    plt.savefig(path_from_home_directory("test/graphs/igrad_social_forces_runtime.png", make_dir=False))
    plt.close()
