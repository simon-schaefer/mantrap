from abc import abstractmethod
from copy import deepcopy
import logging
from typing import Tuple, Union

import numpy as np

from mantrap.constants import planning_horizon_default
from mantrap.simulation.abstract import Simulation


class Solver:
    def __init__(self, sim: Simulation, goal: np.ndarray, planning_horizon: int = planning_horizon_default):
        self._env = sim
        self._goal = goal
        self._planning_horizon = planning_horizon

    def solve(self) -> Tuple[Union[np.ndarray, None], np.ndarray]:
        """Solve the posed solver i.e. find a feasible trajectory for the ego from its initial to its goal state.
        :returns derived ego trajectory or None (no feasible solution) and according predicted ado trajectories
        """
        solver_env = deepcopy(self._env)
        traj_opt = np.zeros((self._planning_horizon, 6))
        traj_opt[0, :] = np.hstack((solver_env.ego.state, solver_env.sim_time))

        ado_trajectories = np.zeros((solver_env.num_ados, solver_env.num_ado_modes, self._planning_horizon, 6))
        for ia, ado in enumerate(solver_env.ados):
            ado_trajectories[ia, :, 0, :] = np.hstack((ado.state, solver_env.sim_time))

        logging.debug(f"Starting trajectory optimization solving for {self._planning_horizon} steps ...")
        for k in range(self._planning_horizon - 1):
            ego_action = self._determine_ego_action(env=solver_env, k=k, traj_opt=traj_opt)

            # Forward simulate environment.
            ado_traj, ego_state = solver_env.step(ego_policy=ego_action)
            ado_trajectories[:, :, k + 1, :] = ado_traj[:, :, 0, :]
            traj_opt[k + 1, :] = ego_state

        logging.debug(f"Finishing up trajectory optimization solving")
        return traj_opt, ado_trajectories

    @abstractmethod
    def _determine_ego_action(self, env: Simulation, k: int, traj_opt: np.ndarray) -> np.ndarray:
        """Determine the next ego action for some time-step k given the previous trajectory traj_opt[:k, :] and
        the simulation environment providing access to all current and previous states.

        :param env:
        :param k:
        :param traj_opt:
        :return:
        """
        pass

    @property
    def environment(self) -> Simulation:
        return self._env

    @property
    def goal(self) -> np.ndarray:
        return self._goal

    @property
    def planning_horizon(self) -> int:
        return self._planning_horizon
