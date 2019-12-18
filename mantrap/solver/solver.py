from abc import abstractmethod
from copy import deepcopy
import logging
from typing import Tuple, Union

import numpy as np

from mantrap.constants import solver_planning_steps, solver_max_steps
from mantrap.simulation.simulation import Simulation


class Solver:
    def __init__(self, sim: Simulation, goal: np.ndarray):
        self._env = sim
        self._goal = goal

    def solve(self, planning_horizon: int = solver_planning_steps, max_steps: int = solver_max_steps) -> Tuple[Union[np.ndarray, None], np.ndarray]:
        """Solve the posed solver i.e. find a feasible trajectory for the ego from its initial to its goal state.
        :returns derived ego trajectory or None (no feasible solution) and according predicted ado trajectories
        """
        solver_env = deepcopy(self._env)
        traj_opt = np.zeros((max_steps, 6))
        traj_opt[0, :] = np.hstack((solver_env.ego.state, solver_env.sim_time))

        ado_trajectories = np.zeros((solver_env.num_ados, solver_env.num_ado_modes, max_steps, 6))
        for ia, ado in enumerate(solver_env.ados):
            ado_trajectories[ia, :, 0, :] = np.hstack((ado.state, solver_env.sim_time))

        logging.info(f"Starting trajectory optimization solving for planning horizon {planning_horizon} steps ...")
        for k in range(max_steps - 1):
            logging.info(f"solver @ time-step k = {k}")
            ego_action = self.determine_ego_action(env=solver_env)
            assert ego_action is not None, "solver failed to find a valid solution for next ego action"
            logging.info(f"solver @k={k}: ego action = {ego_action}")

            # Forward simulate environment.
            ado_traj, ego_state = solver_env.step(ego_policy=ego_action)
            ado_trajectories[:, :, k + 1, :] = ado_traj[:, :, 0, :]
            traj_opt[k + 1, :] = ego_state

            if np.linalg.norm(ego_state[:2] - self._goal) < 0.1:
                traj_opt = traj_opt[:k+2, :]
                ado_trajectories = ado_trajectories[:, :, :k+2, :]
                break

        logging.info(f"Finishing up trajectory optimization solving")
        return traj_opt, ado_trajectories

    @abstractmethod
    def determine_ego_action(self, env: Simulation) -> np.ndarray:
        """Determine the next ego action for some time-step k given the previous trajectory traj_opt[:k, :] and
        the simulation environment providing access to all current and previous states. """
        pass

    @property
    def environment(self) -> Simulation:
        return self._env

    @property
    def goal(self) -> np.ndarray:
        return self._goal
