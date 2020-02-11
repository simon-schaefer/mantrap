import logging
from typing import List, Tuple, Union

import numpy as np
import torch

from mantrap.constants import agent_speed_max
from mantrap.simulation.simulation import GraphBasedSimulation
from mantrap.solver.cgrad.modules import solver_module_dict
from mantrap.solver.solver import IPOPTSolver
from mantrap.utility.shaping import check_trajectory_primitives


class CGradSolver(IPOPTSolver):
    def __init__(
        self, sim: GraphBasedSimulation, goal: torch.Tensor, modules: List[Tuple[str, float]] = None, **solver_params
    ):
        super(CGradSolver, self).__init__(sim, goal, **solver_params)

        # The objective function (and its gradient) are packed into modules, for a more compact representation,
        # the ease of switching between different objective functions and to simplify logging and visualization.
        modules = [("goal", 1.0), ("interaction", 1.0)] if modules is None else modules
        assert all([name in solver_module_dict.keys() for name, _ in modules]), "invalid solver module detected"
        assert all([0.0 <= weight for _, weight in modules]), "invalid solver module weight detected"
        module_args = {"horizon": self.T, "env": self._env, "goal": self.goal}
        self._modules = {m: solver_module_dict[m](weight=w, **module_args) for m, w in modules}

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

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, *args):
        if self.is_verbose:
            self._optimization_log["iter_count"].append(iter_count)
            self._optimization_log["obj_overall"].append(obj_value)
            self._optimization_log["inf_primal"].append(inf_pr)
            self._optimization_log["grad_lagrange"].append(d_norm)
            self._optimization_log["x"].append(self._x_latest)
            for module in self._modules.values():
                module.logging()

    ###########################################################################
    # Utility #################################################################
    ###########################################################################

    @property
    def num_modules(self) -> int:
        return len(self._modules.keys())

    def log_and_clean_up(self):
        # Transfer module logs to main logging dictionary and clean up modules.
        for module_name, module in self._modules.items():
            obj_log, grad_log = module.logs
            self._optimization_log[f"obj_{module_name}"] = obj_log
            self._optimization_log[f"grad_{module_name}"] = grad_log
            module.clean_up()
        # Do logging and visualization of base class with updated optimization log.
        super(CGradSolver, self).log_and_clean_up()
