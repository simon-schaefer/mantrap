import copy
from typing import Dict, List

import pytest
import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.simulation.simulation import GraphBasedSimulation
from mantrap.simulation import PotentialFieldSimulation, SocialForcesSimulation, Trajectron
from mantrap.utility.maths import Distribution, DirecDelta
from mantrap.utility.primitives import straight_line
from mantrap.utility.shaping import check_trajectories, check_controls, check_weights


class ZeroSimulation(GraphBasedSimulation):

    def predict_w_controls(self, controls: torch.Tensor, return_more: bool = False, **graph_kwargs) -> torch.Tensor:
        t_horizon = controls.shape[0]
        return self.predict_wo_ego(t_horizon, return_more=return_more, **graph_kwargs)

    def predict_wo_ego(self, t_horizon: int, return_more: bool = False, **graph_kwargs) -> torch.Tensor:
        policies = torch.zeros((self.num_ados, self.num_modes, t_horizon, 2))
        ados_ghosts_copy = copy.deepcopy(self.ghosts)

        for t in range(t_horizon):
            for j in range(self.num_ghosts):
                i_ado, i_mode = self.index_ghost_id(ghost_id=self.ghosts[j].id)
                self._ado_ghosts[j].agent.update(action=policies[i_ado, i_mode, t, :], dt=self.dt)

        trajectories = torch.zeros(self.num_ados, self.num_modes, t_horizon, 5)
        for ghost in self.ghosts:
            i_ado, i_mode = self.index_ghost_id(ghost_id=ghost.id)
            trajectories[i_ado, i_mode, :, :] = ghost.agent.history[-t_horizon:, :]
        weights = torch.ones((self.num_ados, self.num_modes))

        assert check_controls(policies, num_ados=self.num_ados, num_modes=self.num_modes, t_horizon=t_horizon)
        assert check_weights(weights, num_ados=self.num_ados, num_modes=self.num_modes)
        assert check_trajectories(trajectories, t_horizon=t_horizon, ados=self.num_ados, modes=self.num_modes)

        self._ado_ghosts = ados_ghosts_copy
        return trajectories if not return_more else (trajectories, policies, weights)

    def add_ado(self, **ado_kwargs):
        super(ZeroSimulation, self).add_ado(type=IntegratorDTAgent, **ado_kwargs)

    def _build_connected_graph(self, t_horizon: int, trajectory: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        graph = {}
        for t in range(t_horizon):
            graph.update(self.write_state_to_graph(ego_state=trajectory[t, :], k=t))
        return graph

###########################################################################
# Tests - All Simulations #################################################
###########################################################################
# In order to test the functionality of the simulation environment in a standardized way
@pytest.mark.parametrize(
    "simulation_class", (ZeroSimulation, SocialForcesSimulation, PotentialFieldSimulation, Trajectron)
)
class TestSimulation:

    @staticmethod
    def test_initialization(simulation_class: GraphBasedSimulation.__class__):
        sim = simulation_class(ego_type=IntegratorDTAgent, ego_kwargs={"position": torch.tensor([4, 6])})
        assert torch.all(torch.eq(sim.ego.position, torch.tensor([4, 6]).float()))
        assert sim.num_ados == 0
        assert sim.sim_time == 0.0
        sim.add_ado(position=torch.tensor([6, 7]), velocity=torch.zeros(2), num_modes=1)

        assert torch.all(torch.eq(sim.ghosts[0].agent.position, torch.tensor([6, 7]).float()))
        assert torch.all(torch.eq(sim.ghosts[0].agent.velocity, torch.zeros(2)))

    @staticmethod
    def test_step(simulation_class: GraphBasedSimulation.__class__):
        ado_init_position = torch.zeros(2)
        ado_init_velocity = torch.zeros(2)
        ego_init_position = torch.tensor([-4, 6])
        sim = simulation_class(ego_type=IntegratorDTAgent, ego_kwargs={"position": ego_init_position})

        sim.add_ado(position=ado_init_position, velocity=ado_init_velocity)
        assert sim.num_ados == 1
        assert sim.num_modes == 1
        assert sim.num_ghosts == 1

        t_horizon = 5
        ego_controls = torch.stack([torch.tensor([1, 0])] * t_horizon)
        ego_trajectory = sim.ego.unroll_trajectory(controls=ego_controls, dt=sim.dt)
        for t in range(t_horizon):
            ado_t, ego_t = sim.step(ego_control=ego_controls[t:t+1])

            # Check dimensions of outputted ado and ego states.
            assert ado_t.numel() == 5
            assert ado_t.shape == (1, 1, 1, 5)
            assert ego_t.numel() == 5

            # While the exact value of the ado agent's states depends on the simulation dynamics used, all of them
            # are based on the ego state (control), which is thought to be enforced while forwarding the simulation.
            assert all(torch.eq(ego_t, ego_trajectory[t+1, :]))

    @staticmethod
    def test_step_reset(simulation_class: GraphBasedSimulation.__class__):
        sim = simulation_class(ego_type=IntegratorDTAgent, ego_kwargs={"position": torch.tensor([-4, 6])})
        sim.add_ado(position=torch.zeros(2), velocity=torch.zeros(2), num_modes=1)
        sim.add_ado(position=torch.ones(2), velocity=torch.zeros(2), num_modes=1)

        ego_next_state = torch.rand(5)
        ado_next_states = torch.rand(sim.num_ados, 1, 1, 5)
        sim.step_reset(ego_state_next=ego_next_state, ado_states_next=ado_next_states)

        assert torch.all(torch.eq(sim.ego.state_with_time, ego_next_state))
        for i in range(sim.num_ghosts):
            assert torch.all(torch.eq(sim.ghosts[i].agent.state_with_time, ado_next_states[i]))

    @staticmethod
    def test_prediction_trajectories_shape(simulation_class: GraphBasedSimulation.__class__):
        num_modes = 2
        t_horizon = 4

        sim = simulation_class()
        history = torch.stack(5 * [torch.tensor([1, 0, 0, 0, 0])])
        sim.add_ado(goal=torch.ones(2), position=torch.tensor([-1, 0]), num_modes=num_modes, history=history)
        sim.add_ado(goal=torch.zeros(2), position=torch.tensor([1, 0]), num_modes=num_modes, history=history)

        ado_trajectories = sim.predict_wo_ego(t_horizon=t_horizon)
        print(ado_trajectories.shape)
        assert check_trajectories(ado_trajectories, t_horizon=t_horizon, modes=num_modes, ados=2)

    @staticmethod
    def test_build_connected_graph(simulation_class: GraphBasedSimulation.__class__):
        sim = simulation_class(ego_type=IntegratorDTAgent, ego_kwargs={"position": torch.tensor([-5, 0])})
        sim.add_ado(position=torch.tensor([3, 0]), velocity=torch.zeros(2), goal=torch.tensor([-4, 0]), num_modes=2)
        sim.add_ado(position=torch.tensor([5, 0]), velocity=torch.zeros(2), goal=torch.tensor([-4, 0]), num_modes=2)
        sim.add_ado(position=torch.tensor([10, 0]), velocity=torch.zeros(2), goal=torch.tensor([-4, 0]), num_modes=2)

        prediction_horizon = 10
        trajectory = torch.zeros((prediction_horizon, 4))  # does not matter here anyway
        graphs = sim.build_connected_graph(trajectory=trajectory)

        assert all([f"ego_{k}_position" in graphs.keys() for k in range(prediction_horizon)])
        assert all([f"ego_{k}_velocity" in graphs.keys() for k in range(prediction_horizon)])

    @staticmethod
    def test_ego_graph_updates(simulation_class: GraphBasedSimulation.__class__):
        position = torch.tensor([-5, 0])
        goal = torch.tensor([5, 0])

        sim = simulation_class(ego_type=IntegratorDTAgent, ego_kwargs={"position": position, "velocity": torch.zeros(2)})
        path = straight_line(start_pos=position, end_pos=goal, steps=11)
        velocities = torch.zeros((11, 2))

        graphs = sim.build_connected_graph(trajectory=torch.cat((path, velocities), dim=1))
        for k in range(path.shape[0]):
            assert torch.all(torch.eq(path[k, :], graphs[f"ego_{k}_position"]))

    @staticmethod
    def test_ghost_sorting(simulation_class: GraphBasedSimulation.__class__):
        sim = simulation_class()
        weights_initial = [0.08, 0.1, 0.8, 0.02]
        sim.add_ado(position=torch.zeros(2), num_modes=4, weights=torch.tensor(weights_initial))

        ghost_weights = [ghost.weight for ghost in sim.ghosts]
        assert ghost_weights == list(reversed(sorted(weights_initial)))  # sorted increasing values per default


###########################################################################
# Test - Social Forces Simulation #########################################
###########################################################################
@pytest.mark.parametrize("goal_position", [torch.tensor([2.0, 2.0]), torch.tensor([0.0, -2.0])])
def test_sf_single_ado_prediction(goal_position: torch.Tensor):
    sim = SocialForcesSimulation()
    sim.add_ado(goal=goal_position, position=torch.tensor([-1, -5]), velocity=torch.ones(2) * 0.8, num_modes=1)

    trajectory = torch.squeeze(sim.predict_wo_ego(t_horizon=100))
    assert torch.isclose(trajectory[-1][0], goal_position[0], atol=0.5)
    assert torch.isclose(trajectory[-1][1], goal_position[1], atol=0.5)


def test_sf_static_ado_pair_prediction():
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
def test_sf_ado_ghosts_construction(pos: torch.Tensor, vel: torch.Tensor, num_modes: int, v0s: List[Distribution]):
    sim = SocialForcesSimulation()
    sim.add_ado(goal=torch.zeros(2), position=pos, velocity=vel, num_modes=num_modes, v0s=v0s)

    assert sim.num_modes == num_modes
    assert all([sim.split_ghost_id(ghost.id)[0] == sim.ado_ids[0] for ghost in sim.ghosts])
    assert len(sim.ghosts) == num_modes

    assert all([type(v0) == DirecDelta for v0 in v0s])  # otherwise hard to compare due to sampling
    sim_v0s = [ghost.v0 for ghost in sim.ghosts]
    sim_v0s_exp = [v0.mean for v0 in v0s]
    assert set(sim_v0s) == set(sim_v0s_exp)

    # sim_sigmas = [ghost.sigma for ghost in sim.ado_ghosts]
    # assert np.isclose(np.mean(sim_sigmas), sim_social_forces_defaults["sigma"], atol=0.5)  # Gaussian distributed


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
        forces[i, :] = graph[f"{sim.ghosts[0].id}_0_control"]
        ado_force_norm = graph[f"{sim.ghosts[0].id}_0_output"]
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
