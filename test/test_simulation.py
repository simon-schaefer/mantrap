import copy
from typing import List

import pytest
import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.simulation.simulation import GraphBasedSimulation
from mantrap.simulation import PotentialFieldSimulation, SocialForcesSimulation
from mantrap.utility.maths import Distribution, DirecDelta
from mantrap.utility.primitives import straight_line
from mantrap.utility.shaping import check_trajectories, check_controls, check_weights


class ZeroSimulation(GraphBasedSimulation):
    def __init__(self, num_modes: int = 1, **kwargs):
        super(ZeroSimulation, self).__init__(**kwargs)
        self._num_modes = num_modes

    def predict_w_controls(self, controls: torch.Tensor, return_more: bool = False, **graph_kwargs) -> torch.Tensor:
        t_horizon = controls.shape[0]
        return self.predict_wo_ego(t_horizon, return_more=return_more, **graph_kwargs)

    def predict_wo_ego(self, t_horizon: int, return_more: bool = False, **graph_kwargs) -> torch.Tensor:
        policies = torch.zeros((self.num_ados, self.num_ado_modes, t_horizon, 2))
        ados_ghosts_copy = copy.deepcopy(self.ado_ghosts)

        for t in range(t_horizon):
            for j in range(self.num_ado_ghosts):
                i_ado, i_mode = self.index_ghost_id(ghost_id=self.ado_ghosts[j].id)
                self._ado_ghosts[j].agent.update(action=policies[i_ado, i_mode, t, :], dt=self.dt)

        ghosts = self.ados_most_important_mode()
        trajectories = torch.stack([ghost.agent.history[-t_horizon:, :] for ghost in ghosts]).unsqueeze(1)
        weights = torch.ones((self.num_ados, self.num_ado_modes))

        assert check_controls(policies, num_ados=self.num_ados, num_modes=self.num_ado_modes, t_horizon=t_horizon)
        assert check_weights(weights, num_ados=self.num_ados, num_modes=self.num_ado_modes)
        assert check_trajectories(trajectories, t_horizon=t_horizon, ados=self.num_ados, modes=1)

        self._ado_ghosts = ados_ghosts_copy
        return trajectories if not return_more else (trajectories, policies, weights)

    @property
    def num_ado_modes(self) -> int:
        return self._num_modes


###########################################################################
# Test - Base Simulation ##################################################
###########################################################################
def test_initialization():
    sim = ZeroSimulation(ego_type=IntegratorDTAgent, ego_kwargs={"position": torch.tensor([4, 6])}, dt=1.0)
    assert torch.all(torch.eq(sim.ego.position, torch.tensor([4, 6]).float()))
    assert sim.num_ados == 0
    assert sim.sim_time == 0.0
    sim.add_ado(type=IntegratorDTAgent, position=torch.tensor([6, 7]), velocity=torch.zeros(2), num_modes=1)

    assert torch.all(torch.eq(sim.ado_ghosts[0].agent.position, torch.tensor([6, 7]).float()))
    assert torch.all(torch.eq(sim.ado_ghosts[0].agent.velocity, torch.zeros(2)))


def test_step():
    ado_position = torch.zeros(2)
    ego_position = torch.tensor([-4, 6])
    sim = ZeroSimulation(ego_type=IntegratorDTAgent, ego_kwargs={"position": ego_position}, dt=1.0)

    sim.add_ado(type=IntegratorDTAgent, position=ado_position, velocity=torch.zeros(2))
    assert sim.num_ados == 1
    assert sim.num_ado_modes == 1
    assert sim.num_ado_ghosts == 1

    ego_control = torch.tensor([1, 0])
    num_steps = 100
    ado_trajectory = torch.zeros((sim.num_ados, 1, num_steps, 5))
    ego_trajectory = torch.zeros((num_steps, 5))
    for t in range(num_steps):
        ado_t, ego_t = sim.step(ego_control=ego_control)
        assert ado_t.numel() == 5
        assert ado_t.shape == (1, 1, 1, 5)
        assert ego_t.numel() == 5
        ado_trajectory[:, :, t, :] = ado_t
        ego_trajectory[t, :] = ego_t

    x_start, x_end = ego_position[0].item(), ego_position[0].item() + ego_control[0].item() * sim.dt * num_steps
    y_start, y_end = ego_position[1].item(), ego_position[1].item() + ego_control[1].item() * sim.dt * num_steps
    ego_t_exp = torch.transpose(torch.stack(
        (
            torch.linspace(x_start, x_end, num_steps + 1),
            torch.linspace(y_start, y_end, num_steps + 1),
            torch.ones(num_steps + 1) * ego_control[0],
            torch.ones(num_steps + 1) * ego_control[1],
            torch.linspace(0, num_steps * sim.dt, num_steps + 1),
        )
    ), 0, 1)
    assert torch.all(torch.eq(ego_trajectory, ego_t_exp[1:, :]))


def test_step_reset():
    sim = ZeroSimulation(ego_type=IntegratorDTAgent, ego_kwargs={"position": torch.tensor([-4, 6])}, dt=1.0)
    sim.add_ado(type=IntegratorDTAgent, position=torch.zeros(2), velocity=torch.zeros(2), num_modes=1)
    sim.add_ado(type=IntegratorDTAgent, position=torch.ones(2), velocity=torch.zeros(2), num_modes=1)

    ego_next_state = torch.rand(5)
    ado_next_states = torch.rand(sim.num_ados, 1, 1, 5)
    sim.step_reset(ego_state_next=ego_next_state, ado_states_next=ado_next_states)

    assert torch.all(torch.eq(sim.ego.state_with_time, ego_next_state))
    for i in range(sim.num_ado_ghosts):
        assert torch.all(torch.eq(sim.ado_ghosts[i].agent.state_with_time, ado_next_states[i]))


@pytest.mark.parametrize("position, modes", [(torch.zeros(2), 1), (torch.zeros(2), 4)])
def test_update(position: torch.Tensor, modes: int):
    sim = ZeroSimulation(ego_type=IntegratorDTAgent, ego_kwargs={"position": position}, dt=1, num_modes=modes)
    sim.add_ado(type=IntegratorDTAgent, position=torch.zeros(2), velocity=torch.zeros(2), num_modes=modes)
    sim.add_ado(type=IntegratorDTAgent, position=torch.ones(2), velocity=torch.ones(2), num_modes=modes)
    num_steps = 10
    sim_times = torch.zeros(num_steps)
    ego_positions = torch.zeros((num_steps, 2))
    for t in range(num_steps):
        sim_times[t] = sim.sim_time
        ego_positions[t, :] = sim.ego.position
        sim.step(ego_control=torch.tensor([1, 0]))

    ego_trajectory_x_exp = torch.linspace(position[0].item(), position[0].item() + num_steps - 1, num_steps)
    ego_trajectory_y_exp = torch.linspace(position[1].item(), position[1].item(), num_steps)
    assert torch.all(torch.eq(sim_times, torch.linspace(0, num_steps - 1, num_steps)))
    assert torch.all(torch.eq(ego_positions[:, 0], ego_trajectory_x_exp))
    assert torch.all(torch.eq(ego_positions[:, 1], ego_trajectory_y_exp))


###########################################################################
# Test - Social Forces Simulation #########################################
###########################################################################
@pytest.mark.parametrize("goal_position", [torch.tensor([2.0, 2.0]), torch.tensor([0.0, -2.0])])
def test_single_ado_prediction(goal_position: torch.Tensor):
    sim = SocialForcesSimulation()
    sim.add_ado(goal=goal_position, position=torch.tensor([-1, -5]), velocity=torch.ones(2) * 0.8, num_modes=1)

    trajectory = torch.squeeze(sim.predict_wo_ego(t_horizon=100))
    assert torch.isclose(trajectory[-1][0], goal_position[0], atol=0.5)
    assert torch.isclose(trajectory[-1][1], goal_position[1], atol=0.5)


def test_static_ado_pair_prediction():
    sim = SocialForcesSimulation()
    sim.add_ado(goal=torch.zeros(2), position=torch.tensor([-1, 0]), velocity=torch.tensor([0.1, 0]), num_modes=1)
    sim.add_ado(goal=torch.zeros(2), position=torch.tensor([1, 0]), velocity=torch.tensor([-0.1, 0]), num_modes=1)

    trajectories = sim.predict_wo_ego(t_horizon=100)
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
    assert all([sim.split_ghost_id(ghost.id)[0] == sim.ado_ids[0] for ghost in sim.ado_ghosts])
    assert len(sim.ado_ghosts) == num_modes

    assert all([type(v0) == DirecDelta for v0 in v0s])  # otherwise hard to compare due to sampling
    sim_v0s = [ghost.v0 for ghost in sim.ado_ghosts]
    sim_v0s_exp = [v0.mean for v0 in v0s]
    assert set(sim_v0s) == set(sim_v0s_exp)

    # sim_sigmas = [ghost.sigma for ghost in sim.ado_ghosts]
    # assert np.isclose(np.mean(sim_sigmas), sim_social_forces_defaults["sigma"], atol=0.5)  # Gaussian distributed


@pytest.mark.parametrize("num_modes, t_horizon, v0s", [(2, 4, [DirecDelta(2.3), DirecDelta(1.5)])])
def test_prediction_trajectories_shape(num_modes: int, t_horizon: int, v0s: List[Distribution]):
    sim = SocialForcesSimulation()
    sim.add_ado(goal=torch.ones(2), position=torch.tensor([-1, 0]), num_modes=num_modes, v0s=v0s)
    sim.add_ado(goal=torch.zeros(2), position=torch.tensor([1, 0]), num_modes=num_modes, v0s=v0s)

    ado_trajectories = sim.predict_wo_ego(t_horizon=t_horizon)
    assert check_trajectories(ado_trajectories, t_horizon=t_horizon, modes=num_modes, ados=2)


@pytest.mark.parametrize("num_modes, t_horizon, v0s", [(2, 4, [DirecDelta(2.3), DirecDelta(1.5)])])
def test_prediction_one_agent_only(num_modes: int, t_horizon: int, v0s: List[Distribution]):
    sim = SocialForcesSimulation()
    sim.add_ado(goal=torch.ones(2), position=torch.ones(2), velocity=torch.tensor([1, 0]), num_modes=num_modes, v0s=v0s)

    ado_trajectories = sim.predict_wo_ego(t_horizon=t_horizon)
    assert check_trajectories(ado_trajectories, t_horizon=t_horizon, modes=num_modes, ados=1)
    assert torch.all(torch.eq(ado_trajectories[:, 0, :, :], ado_trajectories[:, 1, :, :]))


def test_build_connected_graph():
    sim = SocialForcesSimulation(IntegratorDTAgent, {"position": torch.tensor([-5, 0])})
    sim.add_ado(position=torch.tensor([3, 0]), velocity=torch.zeros(2), goal=torch.tensor([-4, 0]), num_modes=2)
    sim.add_ado(position=torch.tensor([5, 0]), velocity=torch.zeros(2), goal=torch.tensor([-4, 0]), num_modes=2)
    sim.add_ado(position=torch.tensor([10, 0]), velocity=torch.zeros(2), goal=torch.tensor([-4, 0]), num_modes=2)

    prediction_horizon = 10
    trajectory = torch.zeros((prediction_horizon, 4))  # does not matter here anyway
    graphs = sim.build_connected_graph(ego_trajectory=trajectory)

    assert all([f"ego_{k}_position" in graphs.keys() for k in range(prediction_horizon)])
    assert all([f"ego_{k}_velocity" in graphs.keys() for k in range(prediction_horizon)])


@pytest.mark.parametrize("position, goal", [(torch.tensor([-5, 0]), torch.tensor([5, 0]))])
def test_ego_graph_updates(position: torch.Tensor, goal: torch.Tensor):
    sim = SocialForcesSimulation(IntegratorDTAgent, {"position": position, "velocity": torch.zeros(2)})
    path = straight_line(start_pos=position, end_pos=goal, steps=11)
    velocities = torch.zeros((11, 2))

    graphs = sim.build_connected_graph(ego_trajectory=torch.cat((path, velocities), dim=1))
    for k in range(path.shape[0]):
        assert torch.all(torch.eq(path[k, :], graphs[f"ego_{k}_position"]))


###########################################################################
# Test - PotentialForcesSimulation ########################################
###########################################################################
@pytest.mark.parametrize(
    "pos_1, pos_2",
    [
        (torch.tensor([1, 1]), torch.tensor([2, 2])),
        (torch.tensor([2, 0]), torch.tensor([4, 0])),
        (torch.tensor([0, 2]), torch.tensor([0, 4])),
    ],
)
def test_simplified_sf_simulation(pos_1: torch.Tensor, pos_2: torch.Tensor):
    sim_1 = PotentialFieldSimulation(IntegratorDTAgent, {"position": pos_1})
    sim_2 = PotentialFieldSimulation(IntegratorDTAgent, {"position": pos_2})

    forces = torch.zeros((2, 2))
    gradients = torch.zeros((2, 2))
    for i, sim in enumerate([sim_1, sim_2]):
        sim.add_ado(position=torch.zeros(2))
        graph = sim.build_graph(ego_state=sim.ego.state)
        forces[i, :] = graph[f"{sim.ado_ghosts[0].id}_0_force"]
        ado_force_norm = graph[f"{sim.ado_ghosts[0].id}_0_output"]
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
