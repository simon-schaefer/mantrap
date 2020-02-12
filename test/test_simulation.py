import copy

import pytest
import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.simulation.simulation import Simulation
from mantrap.utility.shaping import check_trajectories, check_policies, check_weights


class ZeroSimulation(Simulation):
    def __init__(self, num_modes: int = 1, **kwargs):
        super(ZeroSimulation, self).__init__(**kwargs)
        self._num_modes = num_modes

    def predict(self, t_horizon: int, ego_trajectory: torch.Tensor = None, verbose: bool = False) -> torch.Tensor:
        policies = torch.zeros((self.num_ados, self.num_ado_modes, t_horizon, 2))
        ados_sim = copy.deepcopy(self._ados)
        for t in range(t_horizon):
            for i in range(self.num_ados):
                for m in range(self.num_ado_modes):
                    ados_sim[i].update(policies[i, m, t, :], dt=self.dt)
        trajectories = torch.stack([ado.history[-t_horizon:, :] for ado in ados_sim]).unsqueeze(1)
        weights = torch.ones((self.num_ados, self.num_ado_modes))

        assert check_policies(policies, num_ados=self.num_ados, num_modes=self.num_ado_modes, t_horizon=t_horizon)
        assert check_weights(weights, num_ados=self.num_ados, num_modes=self.num_ado_modes)
        assert check_trajectories(trajectories, t_horizon=t_horizon, ados=self.num_ados, modes=1)
        return trajectories if not verbose else (trajectories, policies, weights)

    @property
    def num_ado_modes(self) -> int:
        return self._num_modes


###########################################################################
# Tests ###################################################################
###########################################################################


def test_initialization():
    sim = ZeroSimulation(ego_type=IntegratorDTAgent, ego_kwargs={"position": torch.tensor([4, 6])}, dt=1.0)
    assert torch.all(torch.eq(sim.ego.position, torch.tensor([4, 6])))
    assert sim.num_ados == 0
    assert sim.sim_time == 0.0
    sim.add_ado(type=IntegratorDTAgent, position=torch.tensor([6, 7]), velocity=torch.zeros(2))
    assert torch.all(torch.eq(sim.ados[0].position, torch.tensor([6, 7])))
    assert torch.all(torch.eq(sim.ados[0].velocity, torch.zeros(2)))


def test_step():
    ado_position = torch.zeros(2)
    ego_position = torch.tensor([-4, 6])
    sim = ZeroSimulation(ego_type=IntegratorDTAgent, ego_kwargs={"position": ego_position}, dt=1.0)
    sim.add_ado(type=IntegratorDTAgent, position=ado_position, velocity=torch.zeros(2))
    assert sim.num_ados == 1

    ego_policy = torch.tensor([1, 0])
    num_steps = 100
    ado_trajectory = torch.zeros((sim.num_ados, 1, num_steps, 5))
    ego_trajectory = torch.zeros((num_steps, 5))
    for t in range(num_steps):
        ado_t, ego_t = sim.step(ego_policy=ego_policy)
        assert ado_t.numel() == 5
        assert ado_t.shape == (1, 1, 1, 5)
        assert ego_t.numel() == 5
        ado_trajectory[:, :, t, :] = ado_t
        ego_trajectory[t, :] = ego_t

    ego_trajectory_x_exp = torch.linspace(
        ego_position[0].item(), ego_position[0].item() + ego_policy[0].item() * sim.dt * num_steps, num_steps + 1
    )
    ego_trajectory_y_exp = torch.linspace(
        ego_position[1].item(), ego_position[1].item() + ego_policy[1].item() * sim.dt * num_steps, num_steps + 1
    )
    ego_t_exp = torch.stack(
        (
            ego_trajectory_x_exp,
            ego_trajectory_y_exp,
            torch.ones(num_steps + 1) * ego_policy[0],
            torch.ones(num_steps + 1) * ego_policy[1],
            torch.linspace(0, num_steps * sim.dt, num_steps + 1),
        )
    ).T
    assert torch.all(torch.eq(ego_trajectory, ego_t_exp[1:, :]))


@pytest.mark.parametrize("position, modes", [(torch.zeros(2), 1), (torch.zeros(2), 4)])
def test_update(position: torch.Tensor, modes: int):
    sim = ZeroSimulation(ego_type=IntegratorDTAgent, ego_kwargs={"position": position}, dt=1, num_modes=modes)
    sim.add_ado(type=IntegratorDTAgent, position=torch.zeros(2), velocity=torch.zeros(2))
    sim.add_ado(type=IntegratorDTAgent, position=torch.ones(2), velocity=torch.ones(2))
    num_steps = 10
    sim_times = torch.zeros(num_steps)
    ego_positions = torch.zeros((num_steps, 2))
    for t in range(num_steps):
        sim_times[t] = sim.sim_time
        ego_positions[t, :] = sim.ego.position
        sim.step(ego_policy=torch.tensor([1, 0]))

    ego_trajectory_x_exp = torch.linspace(position[0].item(), position[0].item() + num_steps - 1, num_steps)
    ego_trajectory_y_exp = torch.linspace(position[1].item(), position[1].item(), num_steps)
    assert torch.all(torch.eq(sim_times, torch.linspace(0, num_steps - 1, num_steps)))
    assert torch.all(torch.eq(ego_positions[:, 0], ego_trajectory_x_exp))
    assert torch.all(torch.eq(ego_positions[:, 1], ego_trajectory_y_exp))
