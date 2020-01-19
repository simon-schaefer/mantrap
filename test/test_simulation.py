import copy
import time

import numpy as np

from mantrap.agents import IntegratorDTAgent
from mantrap.simulation.simulation import Simulation
from mantrap.utility.shaping import check_ado_trajectories


class ZeroSimulation(Simulation):
    def predict(self, t_horizon: int, ego_trajectory: np.ndarray = None, return_policies: bool = False) -> np.ndarray:
        policies = np.zeros((t_horizon, self.num_ados, 2))
        ados_sim = copy.deepcopy(self._ados)
        for t in range(t_horizon):
            for i in range(self.num_ados):
                ados_sim[i].update(policies[t, i, :], dt=self.dt)
        trajectories = np.expand_dims(np.asarray([ado.history[-t_horizon:, :] for ado in ados_sim]), axis=1)
        assert check_ado_trajectories(trajectories, t_horizon=t_horizon, num_ados=self.num_ados, num_modes=1)
        return trajectories if not return_policies else (trajectories, policies)


def test_initialization():
    sim = ZeroSimulation(IntegratorDTAgent, {"position": np.array([4, 6])}, dt=1.0)
    assert np.array_equal(sim.ego.position, np.array([4, 6]))
    assert sim.num_ados == 0
    assert sim.sim_time == 0.0
    sim.add_ado(type=IntegratorDTAgent, position=np.array([6, 7]), velocity=np.zeros(2))
    assert np.array_equal(sim.ados[0].position, np.array([6, 7]))
    assert np.array_equal(sim.ados[0].velocity, np.zeros(2))


def test_step():
    ado_position = np.zeros(2)
    ego_position = np.array([-4, 6])
    sim = ZeroSimulation(IntegratorDTAgent, {"position": ego_position}, dt=1.0)
    sim.add_ado(type=IntegratorDTAgent, position=ado_position, velocity=np.zeros(2))
    assert sim.num_ados == 1

    ego_policy = np.array([1, 0])
    num_steps = 100
    ado_trajectory = np.zeros((sim.num_ados, 1, num_steps, 6))
    ego_trajectory = np.zeros((num_steps, 6))
    for t in range(num_steps):
        ado_t, ego_t = sim.step(ego_policy=ego_policy)
        assert ado_t.size == 6
        assert ado_t.shape == (1, 1, 1, 6)
        assert ego_t.size == 6
        ado_trajectory[:, :, t, :] = ado_t
        ego_trajectory[t, :] = ego_t

    ego_t_exp = np.vstack(
        (
            np.linspace(ego_position[0], ego_position[0] + ego_policy[0] * sim.dt * num_steps, num_steps + 1),
            np.linspace(ego_position[1], ego_position[1] + ego_policy[1] * sim.dt * num_steps, num_steps + 1),
            np.zeros(num_steps + 1),
            np.ones(num_steps + 1) * ego_policy[0],
            np.ones(num_steps + 1) * ego_policy[1],
            np.linspace(0, num_steps * sim.dt, num_steps + 1),
        )
    ).T
    assert np.array_equal(ego_trajectory, ego_t_exp[1:, :])


def test_reset():
    sim = ZeroSimulation(IntegratorDTAgent, {"position": np.array([1, 5]), "velocity": np.ones(2) * 2}, dt=1.0)
    sim.add_ado(type=IntegratorDTAgent, position=np.array([6, 7]), velocity=np.zeros(2))
    ego_state_new = np.array([5, 1, 0, 2, 0])
    ado_state_new = np.reshape(np.array([0, 0, 0, 5, 7]), (1, 5))
    sim.reset(ego_state=ego_state_new, ado_states=ado_state_new)
    assert np.array_equal(sim.ego.position, ego_state_new[0:2])
    assert np.array_equal(sim.ego.velocity, ego_state_new[3:5])
    assert np.array_equal(sim.ados[0].position, ado_state_new[0, 0:2])
    assert np.array_equal(sim.ados[0].velocity, ado_state_new[0, 3:5])


def test_update():
    sim = ZeroSimulation(ego_type=IntegratorDTAgent, ego_kwargs={"position": np.zeros(2)}, dt=1)
    sim.add_ado(type=IntegratorDTAgent, position=np.zeros(2), velocity=np.zeros(2))
    num_steps = 10
    sim_times = np.zeros(num_steps)
    ego_positions = np.zeros((num_steps, 2))
    for t in range(num_steps):
        sim_times[t] = sim.sim_time
        ego_positions[t, :] = sim.ego.position
        sim.step(ego_policy=np.array([1, 0]))
    assert np.array_equal(sim_times, np.linspace(0, num_steps - 1, num_steps))
    assert np.array_equal(ego_positions[:, 0], np.linspace(0, num_steps - 1, num_steps))
    assert np.array_equal(ego_positions[:, 1], np.zeros(num_steps))
