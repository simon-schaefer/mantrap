from abc import abstractmethod
import logging
from typing import Tuple

import torch

from mantrap.constants import solver_horizon
from mantrap.simulation.simulation import Simulation
from mantrap.utility.utility import expand_state_vector


class Solver:

    def __init__(self, sim: Simulation, goal: torch.Tensor, **solver_params):
        self._env = sim
        self._goal = goal.double()

        # Dictionary of solver parameters.
        self._solver_params = solver_params
        if "planning_horizon" not in self._solver_params.keys():
            self._solver_params["planning_horizon"] = solver_horizon
        if "verbose" not in self._solver_params.keys():
            self._solver_params["verbose"] = False

        assert self._solver_params["planning_horizon"] > 2, "planning horizon must be larger 2"

    def solve(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find the ego trajectory given the internal simulation with the current scene as initial condition.
        Therefore iteratively solve the problem for the scene at t = t_k, update the scene using the internal simulator
        and the derived ego policy and repeat until t_k = `horizon` or until the goal has been reached.
        This method changes the internal environment by forward simulating it over the prediction horizon.

        :return: derived ego trajectory [T, 6].
        :return: ado trajectories [num_ados, modes, T, 6] conditioned on the derived ego trajectory.
        """
        horizon = self._solver_params["planning_horizon"]
        traj_opt = torch.zeros((horizon, 6))
        ado_trajectories = torch.zeros((self._env.num_ados, self._env.num_ado_modes, horizon, 6))

        # Initialize trajectories with current state and simulation time.
        traj_opt[0, :] = expand_state_vector(self._env.ego.state, self._env.sim_time)
        for i, ado in enumerate(self._env.ados):
            ado_trajectories[i, :, 0, :] = expand_state_vector(ado.state, self._env.sim_time)

        logging.info(f"Starting trajectory optimization solving for planning horizon {horizon} steps ...")
        for k in range(horizon - 1):
            logging.info(f"solver @ time-step k = {k}")
            ego_action = self._determine_ego_action(env=self._env)
            assert ego_action is not None, "solver failed to find a valid solution for next ego action"
            logging.info(f"solver @k={k}: ego action = {ego_action}")

            # Forward simulate environment.
            ado_traj, ego_state = self._env.step(ego_policy=ego_action)
            ado_trajectories[:, :, k + 1, :] = ado_traj[:, :, 0, :]
            traj_opt[k + 1, :] = ego_state

            # If the goal state has been reached, break the optimization loop (and shorten trajectories to
            # contain only states up to now (i.e. k + 2 optimization steps instead of max_steps).
            if torch.norm(ego_state[:2] - self._goal) < 0.1:
                traj_opt = traj_opt[: k + 2, :]
                ado_trajectories = ado_trajectories[:, :, : k + 2, :]
                break

        logging.info(f"Finishing up trajectory optimization solving")
        return traj_opt, ado_trajectories

    @abstractmethod
    def _determine_ego_action(self, env: Simulation) -> torch.Tensor:
        """Determine the next ego action for some time-step k given the previous trajectory traj_opt[:k, :] and
        the simulation environment providing access to all current and previous states. """
        pass

    ###########################################################################
    # Solver parameters #######################################################
    ###########################################################################

    @property
    def env(self) -> Simulation:
        return self._env

    @property
    def goal(self) -> torch.Tensor:
        return self._goal

    @property
    def planning_horizon(self) -> int:
        return self._solver_params["planning_horizon"]

    @property
    def is_verbose(self) -> bool:
        return self._solver_params["verbose"]
