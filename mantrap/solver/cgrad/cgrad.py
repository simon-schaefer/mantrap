from collections import defaultdict, deque
import logging
from typing import List, Tuple, Union

import ipopt
import numpy as np
import torch

from mantrap.constants import agent_speed_max, igrad_max_solver_steps, igrad_max_solver_cpu_time
from mantrap.simulation.simulation import GraphBasedSimulation
from mantrap.solver.cgrad.modules import solver_module_dict
from mantrap.solver.solver import Solver
from mantrap.utility.io import build_output_path
from mantrap.utility.shaping import check_trajectory_primitives


class CGradSolver(Solver):
    def __init__(
        self, sim: GraphBasedSimulation, goal: torch.Tensor, modules: List[Tuple[str, float]] = None, **solver_params
    ):
        super(CGradSolver, self).__init__(sim, goal, **solver_params)

        # The objective function (and its gradient) are packed into modules, for a more compact representation,
        # the ease of switching between different objective functions and to simplify logging and visualization.
        modules = [("goal", 0.2), ("interaction", 0.8)] if modules is None else modules
        assert all([name in solver_module_dict.keys() for name, _ in modules]), "invalid solver module detected"
        assert all([0.0 <= weight for _, weight in modules]), "invalid solver module weight detected"
        module_args = {"horizon": self.T, "env": self._env, "goal": self.goal}
        self._modules = {m: solver_module_dict[m](weight=w, **module_args) for m, w in modules}

        # Logging variables. Using default-dict(deque) whenever a new entry is created, it does not have to be checked
        # whether the related key is already existing, since if it is not existing, it is created with a queue as
        # starting value, to which the new entry is appended. With an appending complexity O(1) instead of O(N) the
        # deque is way more efficient than the list type for storing simple floating point numbers in a sequence.
        self._optimization_log = defaultdict(deque) if self.is_verbose else None
        self._x_latest = None  # iteration() function does not input x (!)

    def _determine_ego_action(self, env: GraphBasedSimulation) -> torch.Tensor:
        raise NotImplementedError

    def _solve_optimization(
        self,
        x0: torch.Tensor,
        max_iter: int = igrad_max_solver_steps,
        max_cpu_time: float = igrad_max_solver_cpu_time,
        approx_jacobian: bool = False,
        approx_hessian: bool = True,
        check_derivative: bool = False,
    ):
        """Solve optimization problem by finding constraint bounds, constructing ipopt optimization problem and
        solve it using the parameters defined in the function header."""
        assert check_trajectory_primitives(x0, t_horizon=self.T), "x0 should be ego trajectory"

        # Formulate optimization problem as in standardized IPOPT format.
        x_init, x0_flat = x0[0, :].detach().numpy(), x0.flatten().numpy()
        lb, ub, cl, cu = self.constraint_bounds(x_init=x_init)

        nlp = ipopt.problem(n=2 * self.T, m=len(cl), problem_obj=self, lb=lb, ub=ub, cl=cl, cu=cu)
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
        x_optimized = np.reshape(x_optimized, (self.T, 2))

        # Plot optimization progress.
        self.log_and_clean_up()

        return torch.from_numpy(x_optimized)

    ###########################################################################
    # Optimization formulation ################################################
    # IPOPT requires to use numpy arrays for computation, therefore switch ####
    # everything from torch to numpy here #####################################
    ###########################################################################

    def objective(self, x: np.ndarray) -> float:
        """The objective is to minimize the interaction between the ego and ado, which can be expressed as the
        the L2 difference between the position of every agent with respect to interaction with ego and without
        taking it into account.

        J(x_{ego}) = sum_{t = 0}^T sum_{m = 0}^M || x_m^t(x_{ego}^t) - x_m_{wo}^t
        """
        assert self._env.num_ado_modes == 1, "currently only uni-modal agents are supported"
        x2 = torch.from_numpy(x).view(self.T, 2)
        assert check_trajectory_primitives(x2, t_horizon=self.T), f"x should be ego trajectory with length {self.T}"

        objective = np.sum([m.objective(x2) for m in self._modules.values()])

        logging.debug(f"Objective function = {objective}")
        if self.is_verbose:
            self._x_latest = x.copy()  # logging most current optimization values
        return float(objective)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        assert self._env.num_ado_modes == 1, "currently only uni-modal agents are supported"
        x2 = torch.from_numpy(x).view(self.T, 2)
        assert check_trajectory_primitives(x2, t_horizon=self.T), f"x should be ego trajectory with length {self.T}"

        gradient = np.sum([m.gradient(x2) for m in self._modules.values()], axis=0)

        logging.debug(f"Gradient function = {gradient}")
        if self.is_verbose:
            self._x_latest = x.copy()  # logging most current optimization values
        return gradient

    def constraints(self, x: np.ndarray) -> np.ndarray:
        """An unconstrained optimisation would result in pushing the ego as far as possible from the agents,
        ensuring minimal interaction but also leading to infinite (or at least very large) control effort on the
        one side and not reaching the goal in a reasonable amount of time on the other side. Therefore the trajectory
        length of the ego trajectory is bounded. The most exact but also most computationally expensive way doing
        that is by constraining subsequent trajectory points

        || x_{ego}^t - x_{ego}^{t - 1} || < gamma * dt

        for some maximal velocity gamma and the (simulation) time-step dt. However this would introduce T - 1
        quadratic (at least convex) constraints. Therefore another option would be to set a bound by constraining the
        sum over the whole trajectory only:

        sum_{t = 1}^T || x_{ego}^t - x_{ego}^{t - 1} || < gamma * T
        """
        x2 = np.reshape(x, (self.T, 2))
        inter_path_distance = np.linalg.norm(x2[1:, :] - x2[:-1, :], axis=1)
        initial_position = x2[0, :]

        constraints = np.hstack((inter_path_distance, initial_position))
        logging.debug(f"Constraints vector = {constraints}")
        if self.is_verbose:
            self._x_latest = x.copy()  # logging most current optimization values
        return constraints

    def constraint_bounds(
        self, x_init: np.ndarray
    ) -> Tuple[List[Union[None, float]], List[Union[None, float]], List[Union[None, float]], List[Union[None, float]]]:
        assert x_init.size == 2, "initial position should be two-dimensional"

        # External constraint bounds (inter-point distance and initial point equality).
        cl = [None] * (self.T - 1) + [x_init[0], x_init[1]]
        cu = [agent_speed_max * self._env.dt] * (self.T - 1) + [x_init[0], x_init[1]]

        # Optimization variable bounds.
        lb = (np.ones(2 * self.T) * self._env.axes[0][0]).tolist()
        ub = (np.ones(2 * self.T) * self._env.axes[0][1]).tolist()

        return lb, ub, cl, cu

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        jacobian = np.zeros((2 + (self.T - 1)) * 2 * self.T)  # (2 + self.T - 1) constraints, derivative each wrt to x_i
        x2 = np.reshape(x, (self.T, 2))
        diff = x2[1:] - x2[:-1]
        norm = np.linalg.norm(diff, axis=1) + 1e-6  # prevent zero division

        # inter-point distance constraint jacobian - x and y coordinate.
        for i in range(self.T - 1):
            jacobian[i * 2 * self.T + 2 * i] = -1 / norm[i] * diff[i, 0]
            jacobian[i * 2 * self.T + 2 * i + 1] = -1 / norm[i] * diff[i, 1]
            jacobian[i * 2 * self.T + 2 * (i + 1)] = 1 / norm[i] * diff[i, 0]
            jacobian[i * 2 * self.T + 2 * (i + 1) + 1] = 1 / norm[i] * diff[i, 1]

        # initial position constraint jacobian.
        jacobian[(self.T - 1) * 2 * self.T] = 1
        jacobian[self.T * 2 * self.T + 1] = 1

        logging.debug(f"Constraint jacobian function computed")
        if self.is_verbose:
            self._x_latest = x.copy()  # logging most current optimization values
        return jacobian

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
            for module in self._modules.values():
                module.logging()

    @property
    def T(self) -> int:
        return self.planning_horizon

    @property
    def M(self) -> int:
        return self._env.num_ados

    @property
    def num_modules(self) -> int:
        return len(self._modules.keys())

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

        self._optimization_log = {key: list(values) for key, values in self._optimization_log.items()}

        # Find path to output directory, create it or delete every file inside.
        output_directory_path = build_output_path("test/graphs/igrad_optimization", make_dir=True, free=True)

        # Transfer module logs to main logging dictionary and clean up modules.
        for module_name, module in self._modules.items():
            obj_log, grad_log = module.logs
            self._optimization_log[f"obj_{module_name}"] = obj_log
            self._optimization_log[f"grad_{module_name}"] = grad_log
            module.clean_up()

        # Visualization.
        from mantrap.evaluation.visualization import visualize_optimization
        visualize_optimization(self._optimization_log, env=self._env, dir_path=output_directory_path)

        # Reset optimization logging parameters for next optimization.
        self._optimization_log = defaultdict(deque) if self.is_verbose else None
        self._x_latest = None
