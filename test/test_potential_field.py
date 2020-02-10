import pytest
import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.simulation import PotentialFieldStaticSimulation
from mantrap.utility.io import path_from_home_directory


@pytest.mark.parametrize(
    "pos_1, pos_2",
    [
        (torch.tensor([1, 1]), torch.tensor([2, 2])),
        (torch.tensor([2, 0]), torch.tensor([4, 0])),
        (torch.tensor([0, 2]), torch.tensor([0, 4])),
    ],
)
def test_simplified_sf_simulation(pos_1: torch.Tensor, pos_2: torch.Tensor):
    sim_1 = PotentialFieldStaticSimulation(IntegratorDTAgent, {"position": pos_1})
    sim_2 = PotentialFieldStaticSimulation(IntegratorDTAgent, {"position": pos_2})

    forces = torch.zeros((2, 2))
    gradients = torch.zeros((2, 2))
    for i, sim in enumerate([sim_1, sim_2]):
        sim.add_ado(position=torch.zeros(2))
        graph = sim.build_graph(ego_state=sim.ego.state)
        forces[i, :] = graph[f"{sim.ado_ghosts[0].gid}_0_force"]
        ado_force_norm = graph[f"{sim.ado_ghosts[0].gid}_0_output"]
        gradients[i, :] = torch.autograd.grad(ado_force_norm, graph["ego_0_position"], retain_graph=True)[0].detach()

    # The force is distance based, so more distant agents should affect a smaller force.
    assert torch.norm(forces[0, :]) > torch.norm(forces[1, :])
    assert torch.norm(gradients[0, :]) > torch.norm(gradients[1, :])
    # When the delta position is uni-directional, so e.g. just in x-position, the force as well as the gradient
    # should point only in this direction.
    for i, pos in enumerate([pos_1, pos_2]):
        for k in [0, 1]:
            if pos[k] == 0:
                assert forces[i, k] == gradients[i, k] == 0.0


###########################################################################
# Visualizations ##########################################################
###########################################################################


def visualize_simplified_simulation():
    sim = PotentialFieldStaticSimulation(IntegratorDTAgent, {"position": torch.tensor([-5, 2])}, dt=0.2)
    sim.add_ado(position=torch.zeros(2))

    t_horizon = 100
    ado_states = torch.zeros((sim.num_ados, sim.num_ado_modes, t_horizon, 6))
    ego_states = torch.zeros((t_horizon, 6))
    for t in range(t_horizon):
        ado_states[:, :, t, :], ego_states[t, :] = sim.step(ego_policy=torch.tensor([1, 0]))

    from mantrap.evaluation.visualization import picture_opus

    picture_opus(
        file_path=path_from_home_directory("test/graphs/simplified_simulation"),
        ado_trajectories=ado_states,
        ado_colors=sim.ado_colors,
        ado_ids=sim.ado_ids,
        ego_trajectory=ego_states,
    )
