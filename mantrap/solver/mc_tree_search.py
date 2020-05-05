import time
import typing

import numpy as np
import torch

import mantrap.constants
import mantrap.solver.solver_intermediates


class MonteCarloTreeSearch(mantrap.solver.solver_intermediates.ZControlIntermediate):

    def _optimize(
        self,
        z0: torch.Tensor,
        ado_ids: typing.List[str],
        tag: str = mantrap.constants.TAG_DEFAULT,
        max_iter: int = mantrap.constants.MCTS_MAX_STEPS,
        max_cpu_time: float = mantrap.constants.MCTS_MAX_CPU_TIME,
        **solver_kwargs
    ) -> typing.Tuple[torch.Tensor, float, typing.Dict[str, torch.Tensor]]:
        """Optimization function for single core to find optimal z-vector.

        Given some initial value `z0` find the optimal allocation for z with respect to the internally defined
        objectives and constraints. This function is executed in every thread in parallel, for different initial
        values `z0`. To simplify optimization not all agents in the scene have to be taken into account during
        the optimization but only the ones with ids defined in `ado_ids`.

        MCTS (Monte-Carlo-Tree-Search) uses random sampling during the full allowed computation time and
        returns the trajectory with the best expected objective value.

        :param z0: initial value of optimization variables.
        :param tag: name of optimization call (name of the core).
        :param ado_ids: identifiers of ados that should be taken into account during optimization.
        :returns: z_opt (optimal values of optimization variable vector)
                  objective_opt (optimal objective value)
                  optimization_log (logging dictionary for this optimization = self.log)
        """
        # Find variable bounds for random sampling during search.
        lb, ub = self.optimization_variable_bounds()
        lb, ub = np.asarray(lb), np.asarray(ub)

        # Start stopping conditions (runtime or number of iterations).
        sampling_start_time = time.time()
        sampling_iteration = 0

        # First of all evaluate the default trajectory as "baseline" for further trajectories.
        z_best = z0.detach().numpy()
        obj_best, _ = self._evaluate(z=z_best, tag=tag, ado_ids=ado_ids)

        # Then start sampling (MCTS) loop for finding more optimal trajectories.
        while sampling_iteration < max_iter and (time.time() - sampling_start_time) < max_cpu_time:
            z_sample = np.random.uniform(lb, ub)
            objective, constraint_violation = self._evaluate(z=z_sample, tag=tag, ado_ids=ado_ids)

            if obj_best > objective and constraint_violation < mantrap.constants.SOLVER_CONSTRAINT_LIMIT:
                obj_best = objective
                z_best = z_sample
            sampling_iteration += 1

        # The best sample is re-evaluated for logging purposes, since the last iteration is always assumed to
        # be the best iteration (logging within objective and constraint function).
        self._evaluate(z=z_best, tag=tag, ado_ids=ado_ids)
        return self.z_to_ego_controls(z=z_best), obj_best, self.log

    def _evaluate(self, z: np.ndarray, ado_ids: typing.List[str], tag: str) -> typing.Tuple[float, float]:
        objective = self.objective(z, tag=tag, ado_ids=ado_ids)
        _, constraint_violation = self.constraints(z, ado_ids=ado_ids, return_violation=True, tag=tag)
        return objective, constraint_violation

    ###########################################################################
    # Problem formulation - Objective #########################################
    ###########################################################################
    @staticmethod
    def objective_defaults() -> typing.List[typing.Tuple[str, float]]:
        return [(mantrap.constants.OBJECTIVE_GOAL, 1.0), (mantrap.constants.OBJECTIVE_INTERACTION_POS, 1.0)]

    ###########################################################################
    # Problem formulation - Constraints #######################################
    ###########################################################################
    @staticmethod
    def constraints_defaults() -> typing.List[str]:
        return [mantrap.constants.CONSTRAINT_CONTROL_LIMIT, mantrap.constants.CONSTRAINT_NORM_DISTANCE]

    ###########################################################################
    # Solver properties #######################################################
    ###########################################################################
    @staticmethod
    def solver_name() -> str:
        return "mcts"
