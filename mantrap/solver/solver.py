from abc import abstractmethod
import logging
from typing import Tuple

import torch

from mantrap.constants import solver_horizon
from mantrap.simulation.simulation import Simulation
from mantrap.utility.shaping import check_ego_controls


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
        x5_opt = torch.zeros((horizon, 5))
        ado_trajectories = torch.zeros((horizon - 1, self._env.num_ados, self._env.num_ado_modes, self.T, 5))
        x5_opt_planned = torch.zeros((horizon - 1, self.T, 5))

        # Initialize trajectories with current state and simulation time.
        x5_opt[0, :] = self._env.ego.state_with_time
        ado_trajectories[0] = self.env.predict_wo_ego(t_horizon=self.T).detach()

        logging.info(f"Starting trajectory optimization solving for planning horizon {horizon} steps ...")
        for k in range(horizon - 1):
            logging.info(f"solver @ time-step k = {k}")
            ego_controls = self.determine_ego_controls(**solver_kwargs, iteration_tag=str(k))
            assert check_ego_controls(ego_controls, t_horizon=self.T - 1)
            logging.info(f"solver @k={k}: ego optimized controls = {ego_controls.tolist()}")

            # Forward simulate environment.
            ado_state, ego_state = self._env.step(ego_control=ego_controls[0, :])

            # Logging.
            ado_trajectories[k] = self.env.predict_w_controls(controls=ego_controls).detach()
            x5_opt[k + 1, :] = ego_state.detach()
            x5_opt_planned[k] = self.env.ego.unroll_trajectory(controls=ego_controls, dt=self.env.dt).detach()

            # If the goal state has been reached, break the optimization loop (and shorten trajectories to
            # contain only states up to now (i.e. k + 2 optimization steps instead of max_steps).
            if torch.norm(ego_state[0:2] - self._goal) < 0.1:
                x5_opt = x5_opt[:k + 2, :].detach()
                ado_trajectories = ado_trajectories[:k + 2].detach()
                x5_opt_planned = x5_opt_planned[:k + 2].detach()
                break

        logging.info(f"Finishing up trajectory optimization solving")
        return x5_opt, ado_trajectories, x5_opt_planned

    @abstractmethod
    def determine_ego_controls(self, **solver_kwargs) -> torch.Tensor:
        """Determine the ego control inputs for the internally stated problem and the current state of the environment.
        The implementation crucially depends on the solver class itself and is hence not implemented here.

        :return: ego_controls: control inputs of ego agent for whole planning horizon.
        """
        raise NotImplementedError

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
