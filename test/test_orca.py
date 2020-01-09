import logging
from typing import Dict, Tuple, Union

import numpy as np
import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.simulation.simulation import Simulation
from mantrap.solver import ORCASolver


class ORCASimulation(Simulation):

    orca_rad = 1.0
    orca_dt = 10.0
    sim_dt = 0.25
    sim_speed_max = 4.0

    def __init__(self, ego_type=None, ego_kwargs=None, **kwargs):
        super(ORCASimulation, self).__init__(ego_type, ego_kwargs, dt=self.sim_dt, **kwargs)
        self._ado_goals = []

    def build_graph(self, **kwargs) -> Dict[str, torch.Tensor]:
        pass

    def predict(self, **kwargs) -> np.ndarray:
        pass

    def add_ado(self, goal_position: Union[np.ndarray, None], **ado_kwargs):
        super(ORCASimulation, self)._add_ado(IntegratorDTAgent, log=False, **ado_kwargs)
        self._ado_goals.append(goal_position)

    def step(self, ego_policy: np.ndarray = None) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        self._sim_time = self.sim_time + self.dt

        assert self._ego is None, "simulation merely should have ado agents"
        assert all([ado.__class__ == IntegratorDTAgent for ado in self._ados]), "agents should be single integrators"

        policies = np.zeros((self.num_ados, 2))
        for ia, ado in enumerate(self._ados):
            ado_kwargs = {"position": ado.position, "velocity": ado.velocity, "log": False}
            ado_env = ORCASimulation(IntegratorDTAgent, ego_kwargs=ado_kwargs)
            for other_ado in self._ados[:ia] + self._ados[ia + 1 :]:  # all agent except current loop element
                ado_env.add_ado(position=other_ado.position, velocity=other_ado.velocity, goal_position=None)

            ado_solver = ORCASolver(ado_env, goal=self._ado_goals[ia])
            policies[ia, :] = ado_solver.determine_ego_action(
                ado_env, speed_max=self.sim_speed_max, agent_radius=self.orca_rad, safe_dt=self.orca_dt
            )

        for i, ado in enumerate(self._ados):
            ado.update(policies[i, :], dt=self.dt)
            logging.info(f"simulation @t={self.sim_time} [ado_{ado.id}]: state={ado.state}")
        return np.asarray([ado.state for ado in self._ados]), None


def test_single_agent():
    pos_init = np.zeros(2)
    vel_init = np.zeros(2)
    goal_pos = np.ones(2) * 4

    pos_expected = np.array([[0, 0], [0.707107, 0.707107], [1.41421, 1.41421], [2.12132, 2.12132], [2.82843, 2.82843]])

    sim = ORCASimulation()
    sim.add_ado(position=pos_init, velocity=vel_init, goal_position=goal_pos)

    assert sim.num_ados == 1
    assert np.array_equal(sim.ados[0].position, pos_init)
    assert np.array_equal(sim.ados[0].velocity, vel_init)

    pos = np.zeros(pos_expected.shape)

    pos[0, :] = sim.ados[0].position
    for k in range(1, pos.shape[0]):
        state_k, _ = sim.step()
        pos[k, :] = state_k[0, :2]

    assert np.isclose(np.linalg.norm(pos - pos_expected), 0.0, atol=0.1)


def test_two_agents():
    pos_init = np.array([[-5, 0.1], [5, -0.1]])
    vel_init = np.zeros((2, 2))
    goal_pos = np.array([[5, 0], [-5, 0]])

    pos_expected = np.array(
        [
            [
                [-5, 0.1],
                [-4.8998, 0.107995],
                [-4.63883, 0.451667],
                [-3.65957, 0.568928],
                [-2.68357, 0.6858],
                [-1.7121, 0.802128],
                [-0.747214, 0.917669],
                [0.207704, 1.03202],
                [1.18529, 0.821493],
                [2.16288, 0.61097],
            ],
            [
                [5, -0.1],
                [4.8998, -0.107995],
                [4.63883, -0.451667],
                [3.65957, -0.568928],
                [2.68357, -0.6858],
                [1.7121, -0.802128],
                [0.747214, -0.917669],
                [-0.207704, -1.03202],
                [-1.18529, -0.821493],
                [-2.16288, -0.61097],
            ],
        ]
    )

    sim = ORCASimulation()
    sim.add_ado(position=pos_init[0, :], velocity=vel_init[0, :], goal_position=goal_pos[0, :])
    sim.add_ado(position=pos_init[1, :], velocity=vel_init[1, :], goal_position=goal_pos[1, :])

    pos = np.zeros(pos_expected.shape)
    pos[0, 0, :] = sim.ados[0].position
    pos[1, 0, :] = sim.ados[1].position
    for k in range(1, pos.shape[1]):
        state_k, _ = sim.step()
        pos[:, k, :] = state_k[:, :2]

    assert np.isclose(np.linalg.norm(pos - pos_expected), 0.0, atol=0.1)

#
# def visualize_orca_testing():
#     scenario_func = scenarios.scenario_sf_ego_moving_many_ados
#     sim, goal = scenario_func(sim_type=SocialForcesSimulation, ego_type=IntegratorDTAgent)
#     solver = ORCASolver(sim, goal=goal)
#     ego_traj, ados_traj = solver.solve()
#
#     test_name = f"{ORCASolver.__name__}_{scenario_func.__name__}"
#     evaluate(test_name, ego_traj, ados_traj, sim, goal, do_visualization=True)


if __name__ == "__main__":
    test_two_agents()
