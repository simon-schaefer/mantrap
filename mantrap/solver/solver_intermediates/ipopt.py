import logging
from abc import ABC
from typing import Dict, List, Tuple

import ipopt
import numpy as np
import torch

from mantrap.constants import *
from mantrap.solver.solver import Solver


class IPOPTIntermediate(Solver, ABC):

    def _optimize(
        self,
        z0: torch.Tensor,
        ado_ids: List[str],
        tag: str = "opt",
        max_iter: int = IPOPT_MAX_STEPS_DEFAULT,
        max_cpu_time: float = IPOPT_MAX_CPU_TIME_DEFAULT,
        approx_jacobian: bool = False,
        approx_hessian: bool = True,
        check_derivative: bool = False,
        **solver_kwargs
    ) -> Tuple[torch.Tensor, float, Dict[str, torch.Tensor]]:
        """Optimization function for single core to find optimal z-vector.

        Given some initial value `z0` find the optimal allocation for z with respect to the internally defined
        objectives and constraints. This function is executed in every thread in parallel, for different initial
        values `z0`. To simplify optimization not all agents in the scene have to be taken into account during
        the optimization but only the ones with ids defined in `ado_ids`.

        IPOPT-Solver poses the optimization problem as Non-Linear Program (NLP) and uses the non-linear optimization
        library IPOPT (with Mumps backend) to solve it.

        :param z0: initial value of optimization variables.
        :param tag: name of optimization call (name of the core).
        :param ado_ids: identifiers of ados that should be taken into account during optimization.
        :returns: z_opt (optimal values of optimization variable vector)
                  objective_opt (optimal objective value)
                  optimization_log (logging dictionary for this optimization = self.log)
        """
        # Clean up & detaching graph for deleting previous gradients.
        self._goal = self._goal.detach()
        self._env.detach()

        # Build constraint boundary values (optimisation variables + constraints). The number of constraints
        # depends on the filter, that was selected, as it (might) result in a few number of other agents in
        # the "optimization scene", especially it might lead to zero agents (so an interactively) unconstrained
        # optimization.
        lb, ub = self.optimization_variable_bounds()
        cl, cu = list(), list()
        for name, constraint in self.constraint_module_dict.items():
            lower, upper = constraint.constraint_boundaries(ado_ids=ado_ids)
            cl += list(lower)
            cu += list(upper)
            logging.debug(f"Constraint {name} has bounds lower = {lower} & upper = {upper}")

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
        nlp.addOption("print_level", 5 if self.verbose >= 1 or check_derivative else 0)
        if self.verbose >= 1:
            nlp.addOption("print_timing_statistics", "yes")
        if check_derivative:
            nlp.addOption("derivative_test", "first-order")
            nlp.addOption("derivative_test_tol", 1e-4)

        # Solve optimization problem for "optimal" ego trajectory `x_optimized`.
        z_opt, info = nlp.solve(z0_flat)
        z2_opt = torch.from_numpy(z_opt).view(-1, 2)
        objective_opt = self.objective(z_opt, tag=tag)
        return z2_opt, objective_opt, self.log

    ###########################################################################
    # Optimization formulation - Objective ####################################s
    ###########################################################################
    def gradient(self, z: np.ndarray, ado_ids: List[str] = None, tag: str = TAG_DEFAULT) -> np.ndarray:
        ego_trajectory, grad_wrt = self.z_to_ego_trajectory(z, return_leaf=True)
        gradient = [m.gradient(ego_trajectory, grad_wrt=grad_wrt, ado_ids=ado_ids) for m in self.objective_modules]
        gradient = np.sum(gradient, axis=0)

        logging.debug(f"Gradient function = {gradient}")
        self.log_append(grad_overall=np.linalg.norm(gradient), tag=tag)
        module_log = {f"{LK_GRADIENT}_{key}": mod.grad_current for key, mod in self._objective_modules.items()}
        self.log_append(**module_log, tag=tag)
        return gradient

    ###########################################################################
    # Optimization formulation - Constraints ##################################
    ###########################################################################
    def jacobian(self, z: np.ndarray, ado_ids: List[str] = None, tag: str = TAG_DEFAULT) -> np.ndarray:
        if self.is_unconstrained:
            return np.array([])

        ego_trajectory, grad_wrt = self.z_to_ego_trajectory(z, return_leaf=True)
        jacobian = [m.jacobian(ego_trajectory, grad_wrt=grad_wrt, ado_ids=ado_ids) for m in self.constraint_modules]
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
    def log_keys_performance(self) -> List[str]:
        log_keys = super(IPOPTIntermediate, self).log_keys_performance()
        gradient_keys = [f"{tag}/{LK_GRADIENT}_{key}" for key in self.objective_names for tag in self.cores]
        return log_keys + gradient_keys


###########################################################################
# IPOPT Problem Definition ################################################
###########################################################################
class IPOPTProblem:

    def __init__(self, problem: IPOPTIntermediate, ado_ids: List[str], tag: str = TAG_DEFAULT):
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
