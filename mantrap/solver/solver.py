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
        if "T" not in self._solver_params.keys():
            self._solver_params["T"] = solver_horizon
        if "verbose" not in self._solver_params.keys():
            self._solver_params["verbose"] = False

        assert self._solver_params["T"] > 2, "planning horizon must be larger 2 time-steps due to auto-grad structure"

    def solve(self, horizon: int, **solver_kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find the ego trajectory given the internal simulation with the current scene as initial condition.
        Therefore iteratively solve the problem for the scene at t = t_k, update the scene using the internal simulator
        and the derived ego policy and repeat until t_k = `horizon` or until the goal has been reached.
        This method changes the internal environment by forward simulating it over the prediction horizon.

        :return: derived ego trajectory [horizon, 5].
        :return: ado trajectories [horizon, num_ados, modes, T, 5] conditioned on the derived ego trajectory
        :return: planned ego trajectory for every step [horizon, T, 2].
        """
        x_opt = torch.zeros((horizon, 5))
        ado_trajectories = torch.zeros((horizon - 1, self._env.num_ados, self._env.num_ado_modes, self.T, 5))
        x_opt_planned = torch.zeros((horizon - 1, self.T, 2))

        # Initialize trajectories with current state and simulation time.
        x_opt[0, :] = expand_state_vector(self._env.ego.state, self._env.sim_time)
        ado_trajectories[0] = self.env.predict_wo_ego(t_horizon=self.T).detach()

        logging.info(f"Starting trajectory optimization solving for planning horizon {horizon} steps ...")
        for k in range(horizon - 1):
            logging.info(f"solver @ time-step k = {k}")
            ego_action, x_planned = self._determine_ego_action(iteration_tag=str(k), **solver_kwargs)
            assert ego_action is not None, "solver failed to find a valid solution for next ego action"
            logging.info(f"solver @k={k}: ego action = {ego_action.tolist()}")

            # Forward simulate environment.
            ado_state, ego_state = self._env.step(ego_policy=ego_action)

            # Logging.
            ado_trajectories[k] = self.env.predict(ego_trajectory=x_planned).detach()
            x_opt[k + 1, :] = ego_state.detach()
            x_opt_planned[k] = x_planned.detach()

            # If the goal state has been reached, break the optimization loop (and shorten trajectories to
            # contain only states up to now (i.e. k + 2 optimization steps instead of max_steps).
            if torch.norm(ego_state[0:2] - self._goal) < 0.1:
                x_opt = x_opt[:k + 2, :].detach()
                ado_trajectories = ado_trajectories[:k + 2].detach()
                x_opt_planned = x_opt_planned[:k + 2].detach()
                break

        logging.info(f"Finishing up trajectory optimization solving")
        return x_opt, ado_trajectories, x_opt_planned

    @abstractmethod
    def _determine_ego_action(self, **solver_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine the ego action (and the res"""
        pass

    ###########################################################################
    # Solver parameters #######################################################
    ###########################################################################

    @property
    def env(self) -> Simulation:
        return self._env

    @env.setter
    def env(self, env: Simulation):
        self._env = env

    @property
    def goal(self) -> torch.Tensor:
        return self._goal

    @property
    def T(self) -> int:
        return self._solver_params["T"]

    @property
    def is_verbose(self) -> bool:
        return self._solver_params["verbose"]
