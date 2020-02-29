from abc import abstractmethod
import logging
from typing import Tuple, Union

import ipopt
import joblib
import numpy as np
import torch

from mantrap.constants import ipopt_max_steps, ipopt_max_cpu_time
from mantrap.simulation.simulation import GraphBasedSimulation
from mantrap.solver.solver import Solver


class IPOPTSolver(Solver):

    def __init__(self, sim: GraphBasedSimulation, goal: torch.Tensor, **solver_kwargs):
        super(IPOPTSolver, self).__init__(sim, goal, **solver_kwargs)
        assert self.T > 2, "planning horizon must be larger 2 time-steps due to auto-grad structure"

    def solve_single_optimization(
        self,
        z0: torch.Tensor = None,
        max_iter: int = ipopt_max_steps,
        max_cpu_time: float = ipopt_max_cpu_time,
        approx_jacobian: bool = False,
        approx_hessian: bool = True,
        check_derivative: bool = False,
        return_controls: bool = False,
        iteration_tag: str = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, float], torch.Tensor]:
        """Solve optimization problem by finding constraint bounds, constructing ipopt optimization problem and
        solve it using the parameters defined in the function header."""
        # Clean up & detaching graph for deleting previous gradients.
        self._goal = self._goal.detach()
        self._env.detach()

        # Build constraint boundary values (optimisation variables + constraints).
        lb, ub = self.optimization_variable_bounds()
        cl, cu = list(), list()
        for name, constraint in self._constraint_modules.items():
            cl += list(constraint.lower)
            cu += list(constraint.upper)
            logging.debug(f"Constraint {name} has bounds lower = {constraint.lower} & upper = {constraint.upper}")

        # Formulate optimization problem as in standardized IPOPT format.
        z0 = z0 if z0 is not None else self.z0s_default()[0]
        z0_flat = z0.flatten().numpy().tolist()
        assert len(z0_flat) == len(lb) == len(ub), f"initial value z0 should be {len(lb)} long"

        nlp = ipopt.problem(n=len(z0_flat), m=len(cl), problem_obj=self, lb=lb, ub=ub, cl=cl, cu=cu)
        nlp.addOption("max_iter", max_iter)
        nlp.addOption("max_cpu_time", max_cpu_time)
        if approx_jacobian:
            nlp.addOption("jacobian_approximation", "finite-difference-values")
        if approx_hessian:
            nlp.addOption("hessian_approximation", "limited-memory")

        print_level = 5 if self.is_verbose or check_derivative else 0  # the larger the value, the more print output.
        nlp.addOption("print_level", print_level)
        if self.is_verbose:
            nlp.addOption("print_timing_statistics", "yes")
        if check_derivative:
            nlp.addOption("derivative_test", "first-order")
            nlp.addOption("derivative_test_tol", 1e-4)

        # Solve optimization problem for "optimal" ego trajectory `x_optimized`.
        z_optimized, info = nlp.solve(z0_flat)
        x5_optimized = self.z_to_ego_trajectory(z_optimized)
        z2_optimized = torch.from_numpy(z_optimized).view(-1, 2)
        objective_optimized = self._optimization_log["obj_overall"][-1]

        # Plot optimization progress.
        self.log_and_clean_up(tag=iteration_tag)
        return x5_optimized if not return_controls else (x5_optimized, z2_optimized, float(objective_optimized))

    def determine_ego_controls(self, **solver_kwargs) -> torch.Tensor:
        logging.info("solver starting ipopt optimization procedure")
        z0s = self.z0s_default()

        def evaluate(i: int) -> Tuple[float, torch.Tensor]:
            solver_kwargs["iteration_tag"] = str(self._iteration) + f"_{i}"
            _, z_opt, obj_opt = self.solve_single_optimization(z0=z0s[i], **solver_kwargs, return_controls=True)
            return obj_opt, z_opt

        results = joblib.Parallel(n_jobs=8)(joblib.delayed(evaluate)(i) for i in range(z0s.shape[0]))
        z_opt_best = results[int(np.argmin([obj for obj, _ in results]))][1]
        return self.z_to_ego_controls(z_opt_best.detach().numpy())

    ###########################################################################
    # Initialization ##########################################################
    ###########################################################################
    @abstractmethod
    def z0s_default(self, just_one: bool = False) -> torch.Tensor:
        raise NotImplementedError

    ###########################################################################
    # Optimization formulation - Objective ####################################
    # IPOPT requires to use numpy arrays for computation, therefore switch ####
    # everything from torch to numpy here #####################################
    ###########################################################################
    def gradient(self, z: np.ndarray) -> np.ndarray:
        x4, grad_wrt = self.z_to_ego_trajectory(z, return_leaf=True)
        gradient = np.sum([m.gradient(x4, grad_wrt=grad_wrt) for m in self._objective_modules.values()], axis=0)

        logging.debug(f"Gradient function = {gradient}")
        return gradient

    ###########################################################################
    # Optimization formulation - Constraints ##################################
    ###########################################################################
    def jacobian(self, z: np.ndarray) -> np.ndarray:
        if self.is_unconstrained:
            return np.array([])

        x4, grad_wrt = self.z_to_ego_trajectory(z, return_leaf=True)
        jacobian = np.concatenate([m.jacobian(x4, grad_wrt=grad_wrt) for m in self._constraint_modules.values()])

        logging.debug(f"Constraint jacobian function computed")
        return jacobian

    # wrong hessian should just affect rate of convergence, not convergence in general
    # (given it is semi-positive definite which is the case for the identity matrix)
    # hessian = np.eye(3*self.O)
    def hessian(self, z, lagrange=None, obj_factor=None) -> np.ndarray:
        raise NotImplementedError

    ###########################################################################
    # Visualization & Logging #################################################
    ###########################################################################
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, *args):
        super(IPOPTSolver, self).intermediate_log(iter_count, obj_value, inf_pr)
        self._optimization_log["grad_lagrange"].append(d_norm)
