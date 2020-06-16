import logging
import abc
import typing

import ipopt
import numpy as np
import torch

import mantrap.constants
import mantrap.solver

from .trajopt import TrajOptSolver


class IPOPTIntermediate(TrajOptSolver, abc.ABC):

    def optimize_core(
        self,
        z0: torch.Tensor,
        ado_ids: typing.List[str],
        tag: str = mantrap.constants.TAG_OPTIMIZATION,
        max_cpu_time: float = mantrap.constants.IPOPT_MAX_CPU_TIME_DEFAULT,
        approx_jacobian: bool = False,
        **solver_kwargs
    ) -> typing.Tuple[torch.Tensor, float, typing.Dict[str, torch.Tensor]]:
        """Optimization function for single core to find optimal z-vector.

        Given some initial value `z0` find the optimal allocation for z with respect to the internally defined
        objectives and constraints. This function is executed in every thread in parallel, for different initial
        values `z0`. To simplify optimization not all agents in the scene have to be taken into account during
        the optimization but only the ones with ids defined in `ado_ids`.

        ATTENTION: Since several `optimize_core()` calls are spawned in parallel, one for every process, but
        originating from the same solver class, the method should be self-contained. Hence, no internal
        variables should be updated, since this would lead to race conditions !

        IPOPT-Solver poses the optimization problem as Non-Linear Program (NLP) and uses the non-linear optimization
        library IPOPT (with Mumps backend) to solve it. Documentation: https://pythonhosted.org/ipopt/reference.html

        :param z0: initial value of optimization variables.
        :param tag: name of optimization call (name of the core).
        :param ado_ids: identifiers of ados that should be taken into account during optimization.
        :param max_cpu_time: maximal cpu time until return.
        :param approx_jacobian: if True automatic approximation of Jacobian based on finite-difference values.
        :returns: z_opt (optimal values of optimization variable vector)
                  objective_opt (optimal objective value)
                  optimization_log (logging dictionary for this optimization = self.log)
        """
        # Clean up & detaching graph for deleting previous gradients.
        self._env.detach()

        # Build constraint boundary values (optimisation variables + constraints). The number of constraints
        # depends on the filter, that was selected, as it (might) result in a few number of other agents in
        # the "optimization scene", especially it might lead to zero agents (so an interactively) unconstrained
        # optimization.
        lb, ub = self.optimization_variable_bounds()
        logging.debug(f"Optimization variable constraint has bounds lower = {lb} & upper = {ub}")
        cl, cu = list(), list()
        for name, module in self.module_dict.items():
            lower, upper = module.constraint_boundaries(ado_ids=ado_ids)
            cl += list(lower)
            cu += list(upper)
            logging.debug(f"Constraint {name} has bounds lower = {lower} & upper = {upper}")

        # Formulate optimization problem as in standardized IPOPT format.
        z0_flat = z0.flatten().numpy().tolist()

        # Create ipopt problem with specific tag.
        problem = IPOPTProblem(self, ado_ids=ado_ids, tag=tag)

        # Use definition above to create IPOPT problem.
        nlp = ipopt.problem(n=len(z0_flat), m=len(cl), problem_obj=problem, lb=lb, ub=ub, cl=cl, cu=cu)
        nlp.addOption("max_cpu_time", max_cpu_time)
        nlp.addOption("tol", mantrap.constants.IPOPT_OPTIMALITY_TOLERANCE)  # tolerance for optimality error
        if approx_jacobian:
            nlp.addOption("jacobian_approximation", mantrap.constants.IPOPT_AUTOMATIC_JACOBIAN)
        # Due to the generalized automatic differentiation through large graphs the computational bottleneck
        # of the underlying approach clearly is computing computing derivatives. While calculating the Hessian
        # theoretically would be possible, it would introduce the need of a huge amount of additional computational
        # effort (squared size of gradient !), therefore it will be approximated automatically when needed.
        nlp.addOption("hessian_approximation", mantrap.constants.IPOPT_AUTOMATIC_HESSIAN)

        # The larger the `print_level` value, the more print output IPOPT will provide.
        nlp.addOption("print_level", 5 if self.logger.is_logging else 0)
        if self.logger.is_logging:
            nlp.addOption("print_timing_statistics", "yes")
            # nlp.addOption("derivative_test", "first-order")
            # nlp.addOption("derivative_test_tol", 1e-4)

        # Solve optimization problem for "optimal" ego trajectory `x_optimized`.
        z_opt, info = nlp.solve(z0_flat)
        nlp.close()

        # Return solution as torch tensor.
        z2_opt = torch.from_numpy(z_opt).view(-1, 2)
        objective_opt = self.objective(z_opt, tag=tag, ado_ids=ado_ids)
        return z2_opt, objective_opt, self.logger.log

    ###########################################################################
    # Optimization formulation - Formulation ##################################
    ###########################################################################
    @staticmethod
    def module_hard() -> typing.Union[typing.List[typing.Tuple], typing.List]:
        """List of "hard" optimization modules (objectives, constraint). Hard modules are used for
        warm-starting the trajectory optimization and should therefore be simple to solve while still
        encoding a good guess of possible solutions.

        The IPOPT solver already uses the optimization variable boundaries as control limit, since we
        optimize for z = controls. Therefore only the goal module is required as a hard module.
        """
        return [mantrap.modules.GoalNormModule, mantrap.modules.SpeedLimitModule]

    ###########################################################################
    # Optimization formulation - Gradient #####################################
    ###########################################################################
    def gradient(self, z: np.ndarray, ado_ids: typing.List[str] = None, tag: str = mantrap.constants.TAG_OPTIMIZATION
                 ) -> np.ndarray:
        """Gradient computation function.

        Compute the objectives gradient for some value of the optimization variable `z` based on the
        gradient implementations of the objective modules. Sum all these gradients together for the
        final gradient estimate.
        """
        ado_ids = ado_ids if ado_ids is not None else self.env.ado_ids
        ego_trajectory, grad_wrt = self.z_to_ego_trajectory(z, return_leaf=True)
        gradient = [m.gradient(ego_trajectory, grad_wrt=grad_wrt, tag=tag, ado_ids=ado_ids) for m in self.modules]
        gradient = np.sum(gradient, axis=0)

        self.logger.log_append(grad_overall=np.linalg.norm(gradient), tag=tag)
        module_log = {f"{mantrap.constants.LT_GRADIENT}_{key}": mod.grad_current(tag=tag)
                      for key, mod in self.module_dict.items()}
        self.logger.log_append(**module_log, tag=tag)
        return gradient

    ###########################################################################
    # Optimization formulation - Jacobian #####################################
    ###########################################################################
    def jacobian(self, z: np.ndarray, ado_ids: typing.List[str] = None, tag: str = mantrap.constants.TAG_OPTIMIZATION
                 ) -> np.ndarray:
        """Jacobian of constraints computation function.

        Compute the constraints jacobian for some value of the optimization variable `z` based on the
        jacobian implementations of the constraints modules. Concatenate all these gradients together
        for the final jacobian estimate.
        """
        ado_ids = ado_ids if ado_ids is not None else self.env.ado_ids
        ego_trajectory, grad_wrt = self.z_to_ego_trajectory(z, return_leaf=True)

        jacobian = [m.jacobian(ego_trajectory, grad_wrt=grad_wrt, tag=tag, ado_ids=ado_ids) for m in self.modules]
        jacobian = [x.flatten() for x in jacobian if x.size > 0]
        jacobian = np.concatenate(jacobian)
        return jacobian

    def jacobian_structure(self, ado_ids: typing.List[str] = None, tag: str = mantrap.constants.TAG_OPTIMIZATION
                           ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Sparsity structure of Jacobian matrix.

        The structure of the Jacobian matrix is defined by the indices of non-zero elements in the Jacobian
        matrix. As it is defined by module, the sparsity structures of all modules are concatenated.
        """
        ado_ids = ado_ids if ado_ids is not None else self.env.ado_ids

        # Get jacobian structures of optimization modules, and filter out the pure objective modules
        # (=> None jacobian return value).
        structures = [m.jacobian_structure(ado_ids=ado_ids, tag=tag) for m in self.modules]
        num_constraints = [m.num_constraints(ado_ids=ado_ids) for m in self.modules]

        # Shift structures using the length of the module-wise jacobian which is equal to the number of
        # constraints times the length of the `grad_wrt` tensor, which is 2 * t_planning.
        structure_full = np.array([])
        shift = 0
        for structure, num_cons in zip(structures, num_constraints):
            if structure is None:
                continue
            structure_shifted = structure + shift  # type: np.ndarray
            structure_full = np.concatenate((structure_full, structure_shifted))
            shift += num_cons * 2 * self.planning_horizon

        structure_full_flat = structure_full.astype(int)
        return np.unravel_index(structure_full_flat, dims=(sum(num_constraints), 2 * self.planning_horizon))

    # wrong hessian should just affect rate of convergence, not convergence in general
    # (given it is semi-positive definite which is the case for the identity matrix)
    # hessian = np.eye(3*self.O)
    # def hessian(self, z, lagrange=None, obj_factor=None) -> np.ndarray:
    #    raise NotImplementedError

    ###########################################################################
    # Visualization & Logging #################################################
    ###########################################################################
    def log_keys_performance(self, tag: str = mantrap.constants.TAG_OPTIMIZATION) -> typing.List[str]:
        log_keys = super(IPOPTIntermediate, self).log_keys_performance()
        gradient_keys = [f"{tag}/{mantrap.constants.LT_GRADIENT}_{key}"
                         for key in self.module_names]
        return log_keys + gradient_keys


###########################################################################
# IPOPT Problem Definition ################################################
###########################################################################
class IPOPTProblem:

    def __init__(
        self,
        problem: IPOPTIntermediate,
        ado_ids: typing.List[str],
        tag: str = mantrap.constants.TAG_OPTIMIZATION
    ):
        self.problem = problem
        self.tag = tag
        self.ado_ids = ado_ids

    def objective(self, z: np.ndarray) -> float:
        return self.problem.objective(z, tag=self.tag, ado_ids=self.ado_ids)

    def gradient(self, z: np.ndarray) -> np.ndarray:
        return self.problem.gradient(z, tag=self.tag, ado_ids=self.ado_ids)

    def constraints(self, z: np.ndarray) -> np.ndarray:
        return self.problem.constraints(z, tag=self.tag, ado_ids=self.ado_ids)

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        return self.problem.jacobian(z, tag=self.tag, ado_ids=self.ado_ids)

    def jacobianstructure(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        return self.problem.jacobian_structure(tag=self.tag, ado_ids=self.ado_ids)

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, *args):
        pass
