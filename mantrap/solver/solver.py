from abc import abstractmethod
from collections import defaultdict, deque
import logging
from typing import List, Tuple

import ipopt
import numpy as np
import torch

from mantrap.constants import solver_horizon, ipopt_max_solver_steps, ipopt_max_solver_cpu_time
from mantrap.simulation.simulation import Simulation
from mantrap.utility.io import build_output_path
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


class IPOPTSolver(Solver):

    def __init__(self, sim: Simulation, goal: torch.Tensor, **solver_params):
        super(IPOPTSolver, self).__init__(sim, goal, **solver_params)

        # Logging variables. Using default-dict(deque) whenever a new entry is created, it does not have to be checked
        # whether the related key is already existing, since if it is not existing, it is created with a queue as
        # starting value, to which the new entry is appended. With an appending complexity O(1) instead of O(N) the
        # deque is way more efficient than the list type for storing simple floating point numbers in a sequence.
        self._optimization_log = defaultdict(deque) if self.is_verbose else None
        self._x_latest = None  # iteration() function does not input x (!)

    def _solve_optimization(
        self,
        x0: torch.Tensor,
        max_iter: int = ipopt_max_solver_steps,
        max_cpu_time: float = ipopt_max_solver_cpu_time,
        approx_jacobian: bool = False,
        approx_hessian: bool = True,
        check_derivative: bool = False,
    ):
        """Solve optimization problem by finding constraint bounds, constructing ipopt optimization problem and
        solve it using the parameters defined in the function header."""
        lb, ub, cl, cu = self.constraint_bounds(x_init=x0)

        # Formulate optimization problem as in standardized IPOPT format.
        x0_flat = x0.flatten().numpy().tolist()
        nlp = ipopt.problem(n=len(x0_flat), m=len(cl), problem_obj=self, lb=lb, ub=ub, cl=cl, cu=cu)
        nlp.addOption("max_iter", max_iter)
        nlp.addOption("max_cpu_time", max_cpu_time)
        if approx_jacobian:
            nlp.addOption("jacobian_approximation", "finite-difference-values")
        if approx_hessian:
            nlp.addOption("hessian_approximation", "limited-memory")
        nlp.addOption("print_level", 5)  # the larger the value, the more print output.
        if self.is_verbose or check_derivative:
            nlp.addOption("print_timing_statistics", "yes")
            nlp.addOption("derivative_test", "first-order")
            nlp.addOption("derivative_test_tol", 1e-4)

        # Solve optimization problem for "optimal" ego trajectory `x_optimized`.
        x_optimized, info = nlp.solve(x0_flat)
        x_optimized = self.x_to_ego_trajectory(x_optimized)

        # Plot optimization progress.
        self.log_and_clean_up()

        return x_optimized

    ###########################################################################
    # Optimization formulation ################################################
    # IPOPT requires to use numpy arrays for computation, therefore switch ####
    # everything from torch to numpy here #####################################
    ###########################################################################

    @abstractmethod
    def objective(self, x: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def constraints(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def constraint_bounds(self, x_init: torch.Tensor) -> Tuple[List, List, List, List]:
        raise NotImplementedError

    @abstractmethod
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # wrong hessian should just affect rate of convergence, not convergence in general
    # (given it is semi-positive definite which is the case for the identity matrix)
    # hessian = np.eye(3*self.O)
    def hessian(self, x, lagrange=None, obj_factor=None) -> np.ndarray:
        raise NotImplementedError

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, *args):
        if self.is_verbose:
            self._optimization_log["iter_count"].append(iter_count)
            self._optimization_log["obj_overall"].append(obj_value)
            self._optimization_log["inf_primal"].append(inf_pr)
            self._optimization_log["grad_lagrange"].append(d_norm)
            self._optimization_log["x"].append(self._x_latest)

    ###########################################################################
    # Utility #################################################################
    ###########################################################################

    @abstractmethod
    def x_to_ego_trajectory(self, x: np.ndarray) -> torch.Tensor:
        raise NotImplementedError

    @property
    def T(self) -> int:
        return self.planning_horizon

    @property
    def M(self) -> int:
        return self._env.num_ados

    ###########################################################################
    # Visualization ###########################################################
    ###########################################################################
    def log_and_clean_up(self):
        """Clean up optimization logs and reset optimization parameters.
        IPOPT determines the CPU time including the intermediate function, therefore if we would plot at every step,
        we would loose valuable optimization time. Therefore the optimization progress is plotted all at once at the
        end of the optimization process."""

        # Plotting only makes sense if you see some progress in the optimization, i.e. compare and figure out what
        # the current optimization step has changed.
        if not self.is_verbose or len(self._optimization_log["iter_count"]) < 2:
            return

        self._optimization_log = {k: list(data) for k, data in self._optimization_log.items() if not type(data) == list}

        # Visualization. Find path to output directory, create it or delete every file inside.
        from mantrap.evaluation.visualization import visualize_optimization
        output_directory_path = build_output_path("test/graphs/igrad_optimization", make_dir=True, free=True)
        visualize_optimization(self._optimization_log, env=self._env, dir_path=output_directory_path)

        # Reset optimization logging parameters for next optimization.
        self._optimization_log = defaultdict(deque) if self.is_verbose else None
        self._x_latest = None
