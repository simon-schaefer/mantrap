import logging
from typing import Dict, Tuple

import ipopt
import numpy as np
import torch

from mantrap.constants import ipopt_max_steps, ipopt_max_cpu_time
from mantrap.solver.solver import Solver


class IPOPTSolver(Solver):

    def _optimize(
        self,
        z0: torch.Tensor,
        ado_ids: List[str] = None,
        tag: str = "core",
        max_iter: int = ipopt_max_steps,
        max_cpu_time: float = ipopt_max_cpu_time,
        approx_jacobian: bool = False,
        approx_hessian: bool = True,
        check_derivative: bool = False,
        **solver_kwargs
    ) -> Tuple[torch.Tensor, float, Dict[str, torch.Tensor]]:
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
        z0_flat = z0.flatten().numpy().tolist()
        assert len(z0_flat) == len(lb) == len(ub), f"initial value z0 should be {len(lb)} long"

        # Create ipopt problem with specific tag.
        problem = IPOPTProblem(self, ado_ids=ado_ids, tag=tag)

        # Use definition above to create IPOPT problem.
        nlp = ipopt.problem(n=len(z0_flat), m=len(cl), problem_obj=problem, lb=lb, ub=ub, cl=cl, cu=cu)
        nlp.addOption("max_iter", max_iter)
        nlp.addOption("max_cpu_time", max_cpu_time)
        if approx_jacobian:
            nlp.addOption("jacobian_approximation", "finite-difference-values")
        if approx_hessian:
            nlp.addOption("hessian_approximation", "limited-memory")

        # The larger the `print_level` value, the more print output IPOPT will provide.
        nlp.addOption("print_level", 5 if self.verbose > 2 or check_derivative else 0)
        if self.verbose > 2:
            nlp.addOption("print_timing_statistics", "yes")
        if check_derivative:
            nlp.addOption("derivative_test", "first-order")
            nlp.addOption("derivative_test_tol", 1e-4)

        # Solve optimization problem for "optimal" ego trajectory `x_optimized`.
        z_opt, info = nlp.solve(z0_flat)
        z2_opt = torch.from_numpy(z_opt).view(-1, 2)
        objective_opt = self.objective(z_opt, tag=tag)
        return z2_opt, objective_opt, self.optimization_log

    ###########################################################################
    # Optimization formulation - Objective ####################################s
    ###########################################################################
    def gradient(self, z: np.ndarray, ado_ids: List[str] = None, tag: str = "core") -> np.ndarray:
        x4, grad_wrt = self.z_to_ego_trajectory(z, return_leaf=True)
        gradient = [m.gradient(x4, grad_wrt=grad_wrt, ado_ids=ado_ids) for m in self._objective_modules.values()]
        gradient = np.sum(gradient, axis=0)

        logging.debug(f"Gradient function = {gradient}")
        self.log_append(grad_overall=gradient, tag=tag)
        self.log_append(**{f"grad_{key}": mod.grad_current for key, mod in self._objective_modules.items()}, tag=tag)
        return gradient

    ###########################################################################
    # Optimization formulation - Constraints ##################################
    ###########################################################################
    def jacobian(self, z: np.ndarray, ado_ids: List[str] = None, tag: str = "core") -> np.ndarray:
        if self.is_unconstrained:
            return np.array([])

        x4, grad_wrt = self.z_to_ego_trajectory(z, return_leaf=True)
        jacobian = [m.jacobian(x4, grad_wrt=grad_wrt, ado_ids=ado_ids) for m in self._constraint_modules.values()]
        jacobian = np.concatenate(jacobian)

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
    def log_reset(self, log_horizon: int):
        super(IPOPTSolver, self).log_reset(log_horizon)
        for tag in self.cores:
            for k in range(log_horizon):
                self._optimization_log.update({f"{tag}/grad_{key}_{k}": [] for key in self.objective_keys})

    def log_summarize(self, tag: str = "core"):
        super(IPOPTSolver, self).log_summarize()

        gradient_keys = [f"{tag}/grad_{key}" for key in self.objective_keys for tag in self.cores]
        for key in gradient_keys:
            # Summarize best gradient values to one website.
            summary = [self._optimization_log[f"{key}_{k}"][-1] for k in range(self._iteration)]
            self._optimization_log[key] = torch.stack(summary)
            # Restructure 1-size tensor to actual vectors (objective and constraint).
            for k in range(self._iteration):
                self._optimization_log[f"{key}_{k}"] = torch.stack(self._optimization_log[f"{key}_{k}"])


###########################################################################
# IPOPT Problem Definition ################################################
###########################################################################
class IPOPTProblem:

    def __init__(self, problem: IPOPTSolver, ado_ids: List[str], tag: str = "core"):
        self.problem = problem
        self.tag = tag
        self.ado_ids = ado_ids

    def objective(self, z: np.ndarray) -> float:
        return self.problem.objective(z, tag=self.tag, ado_ids=self.ado_ids)

    def gradient(self, z: np.ndarray,) -> np.ndarray:
        return self.problem.gradient(z, tag=self.tag, ado_ids=self.ado_ids)

    def constraints(self, z: np.ndarray) -> np.ndarray:
        return self.problem.constraints(z, tag=self.tag, ado_ids=self.ado_ids)

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        return self.problem.jacobian(z, tag=self.tag, ado_ids=self.ado_ids)

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, *args):
        pass
