import abc
import time
import typing

import numpy as np
import torch

import mantrap.constants
import mantrap.utility.shaping

from .trajopt import TrajOptSolver


class SearchIntermediate(TrajOptSolver, abc.ABC):

    ###########################################################################
    # Initialization ##########################################################
    ###########################################################################
    def initialize(self, **solver_params):
        super(SearchIntermediate, self).initialize(**solver_params)
        # Find variable bounds for random sampling during search.
        lb, ub = self.optimization_variable_bounds()
        self._z_bounds = np.asarray(lb), np.asarray(ub)

    ###########################################################################
    # Optimization ############################################################
    ###########################################################################
    def _optimize(
        self,
        z0: torch.Tensor,
        ado_ids: typing.List[str],
        tag: str = mantrap.constants.TAG_OPTIMIZATION,
        max_cpu_time: float = mantrap.constants.SEARCH_MAX_CPU_TIME,
        **solver_kwargs
    ) -> typing.Tuple[torch.Tensor, float, typing.Dict[str, torch.Tensor]]:
        """Optimization function for single core to find optimal z-vector.

        Given some initial value `z0` find the optimal allocation for z with respect to the internally defined
        objectives and constraints. This function is executed in every thread in parallel, for different initial
        values `z0`. To simplify optimization not all agents in the scene have to be taken into account during
        the optimization but only the ones with ids defined in `ado_ids`.

        ATTENTION: Since several `_optimize()` calls are spawned in parallel, one for every process, but
        originating from the same solver class, the method should be self-contained. Hence, no internal
        variables should be updated, since this would lead to race conditions !

        In general searching algorithms (at least the ones implemented within this project) have a similar
        "frame", which is some termination constraint, here the computation runtime, and some inner optimization
        loop which repeats until the algorithm has either terminated or is done.

        :param z0: initial value of optimization variables.
        :param tag: name of optimization call (name of the core).
        :param ado_ids: identifiers of ados that should be taken into account during optimization.
        :returns: z_opt (optimal values of optimization variable vector)
                  objective_opt (optimal objective value)
                  optimization_log (logging dictionary for this optimization = self.log)
        """
        # Start stopping conditions (runtime or number of iterations).
        sampling_start_time = time.time()

        # Then start searching loop for finding more optimal trajectories.
        z_best, obj_best, iteration = None, np.inf, 0
        while (time.time() - sampling_start_time) < max_cpu_time:
            z_best_candidate, obj_best_candidate, iteration_candidate, is_finished = self._optimize_inner(
                z_best, obj_best, iteration, tag, ado_ids)

            # Update iteration variables (z_best, obj_best, iteration) only if the objective has been
            # improved over the iteration.
            if obj_best_candidate <= obj_best:
                z_best = z_best_candidate
                obj_best = obj_best_candidate
                iteration = iteration_candidate

            # If solver claims to be finished, end the iteration before the runtime has exceeded.
            if is_finished:
                break

        # The best sample is re-evaluated for logging purposes, since the last iteration is always assumed to
        # be the best iteration (logging within objective and constraint function).
        self._evaluate(z=z_best, tag=tag, ado_ids=ado_ids)
        ego_controls = self.z_to_ego_controls(z=z_best)
        assert mantrap.utility.shaping.check_ego_controls(ego_controls, t_horizon=self.planning_horizon)
        return ego_controls, obj_best, self.log

    @abc.abstractmethod
    def _optimize_inner(self, z_best: np.ndarray, obj_best: float, iteration: int, tag: str, ado_ids: typing.List[str]
                        ) -> typing.Tuple[np.ndarray, float, int, bool]:
        """Inner optimization/search function.

        ATTENTION: See self-containment comment in `_optimize()` method description.

        :param z_best: best assignment of optimization vector so far.
        :param obj_best: according objective function value.
        :param iteration: current search (outer loop) iteration.
        :param tag: name of optimization call (name of the core).
        :param ado_ids: identifiers of ados that should be taken into account during optimization.
        :returns: updated best z-values, updated best objective, outer loop iteration, termination flag.
        """
        raise NotImplementedError

    def _evaluate(self, z: np.ndarray, ado_ids: typing.List[str], tag: str) -> typing.Tuple[float, float]:
        """Evaluate "value" of z-values, i.e. some choice of optimization variable assignment, by computing
        the overall objective value as well as the constraint violation (aka the feasibility of the choice). """
        objective = self.objective(z, tag=tag, ado_ids=ado_ids)
        _, constraint_violation = self.constraints(z, ado_ids=ado_ids, return_violation=True, tag=tag)
        return objective, constraint_violation

    ###########################################################################
    # Optimization formulation parameters #####################################
    ###########################################################################
    @property
    def z_bounds(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        return self._z_bounds
