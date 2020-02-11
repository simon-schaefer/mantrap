import pytest
import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.simulation import PotentialFieldStaticSimulation
from mantrap.utility.io import build_output_path


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
