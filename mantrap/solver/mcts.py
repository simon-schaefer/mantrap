import typing

import numpy as np

import mantrap.constants
import mantrap.modules

from .base import SearchIntermediate, ZControlIntermediate


class MonteCarloTreeSearch(SearchIntermediate, ZControlIntermediate):

    ###########################################################################
    # Optimization ############################################################
    ###########################################################################
    def _optimize_inner(self, z_best: np.ndarray, obj_best: float, iteration: int, tag: str, ado_ids: typing.List[str],
                        num_breadth_samples: int = mantrap.constants.MCTS_NUMBER_BREADTH_SAMPLES,
                        num_depth_samples: int = mantrap.constants.MCTS_NUMBER_DEPTH_SAMPLES,
                        ) -> typing.Tuple[np.ndarray, float, int, bool]:
        """Inner optimization/search function.

        ATTENTION: See self-containment comment in `_optimize()` method description.

        MCTS (Monte-Carlo-Tree-Search) iteratively optimizes thr trajectory by sampling-based estimating the
        actual cost-to-go assigned to some choice of optimization variable value. Therefore until the end
        of the trajectory, several possible next z-values (i.e. z-values for z_{i+1} in the ith optimization
        step) are sampled and for each sampled value possible trajectories are rolled out. The average over
        all these rollouts for each value z_{i+1} determines its estimated value, while the value for z with
        smallest objective value (while being within the constraint bounds) is chosen. Then the process repeats
        for the values z_{i+2} until z_{n} or until the computation time has exceeded.

        :param z_best: best assignment of optimization vector so far.
        :param obj_best: according objective function value.
        :param iteration: current search (outer loop) iteration.
        :param tag: name of optimization call (name of the core).
        :param ado_ids: identifiers of ados that should be taken into account during optimization.
        :returns: updated best z-values, updated best objective, outer loop iteration, termination flag.
        """
        lb, ub = self.z_bounds
        assert len(lb) == len(ub)
        assert 0 <= iteration <= len(lb)

        # Sample choices of the next z-value to evaluate (initially z_best = None).
        # z_priors = (number of priors, t_horizon, z_size)
        lb_current, ub_current = lb[iteration], ub[iteration]
        samples = np.random.uniform(np.ones(2 * num_breadth_samples) * lb_current,
                                    np.ones(2 * num_breadth_samples) * ub_current).reshape(-1, 1, 2)
        if z_best is not None:
            assert len(z_best) == self.planning_horizon * 2  # assuming z = (-1, 2) = 2D !!
            z_best_repeated = np.stack([z_best[:2*iteration].reshape(-1, 2)] * num_breadth_samples)
            z_priors = np.concatenate((z_best_repeated, samples), axis=1)
        else:
            z_priors = samples

        # Estimate the value of each prior of z by unrolling sampled "complete" assignments.
        z_mean_estimated_values = np.zeros(num_breadth_samples)
        z_best_samples = [(np.array([]), np.inf)] * num_breadth_samples
        max_constrain_violation = mantrap.constants.CONSTRAINT_VIOLATION_PRECISION
        for iz in range(num_breadth_samples):
            zi_prior = z_priors[iz, :, :].flatten()
            values_i = []
            z_i_best = (None, np.inf)
            for _ in range(num_depth_samples):
                zij = np.random.uniform(lb[2*(iteration+1):], ub[2*(iteration+1):])
                zij = np.concatenate((zi_prior, zij))
                assert len(zij) == self.planning_horizon * 2
                obj_ij, violation_ij = self._evaluate(zij, tag=tag, ado_ids=ado_ids)

                # Check whether sample ij is feasible, i.e. constraint violation below some threshold.
                # In case append the value to the objective values for the prior sample i.
                if violation_ij < max_constrain_violation:
                    values_i.append(obj_ij)

                    # Additionally store the best sample of the current z prior. Later on in case the
                    # current prior is the best out of all prior its best sample assignment is returned.
                    if obj_ij < z_i_best[1]:  # z_i_best = (sample, value)
                        z_i_best = (zij, obj_ij)

            # Store the mean (feasible) objective value and the best sample, if any, to compare later on.
            z_mean_estimated_values[iz] = np.mean(values_i) if len(values_i) > 0 else np.inf
            z_best_samples[iz] = z_i_best

        # Now check which prior has been the best prior overall. If the best trajectory belonging to this
        # best prior is valid (i.e. not None), return it regardless of the previous best value, since we
        # could just be lucky with sampling and rather prefer iterative optimization.
        # Termination only if the full planning horizon has been observed.
        iz_best = int(np.argmin(z_mean_estimated_values))
        z_best_candidate, obj_best_candidate = z_best_samples[iz_best]
        should_terminate = iteration + 1 == self.planning_horizon
        if z_best_candidate is not None:
            return z_best_candidate, float(obj_best_candidate), iteration + 1, should_terminate
        else:
            return z_best, obj_best, iteration, should_terminate

    ###########################################################################
    # Optimization formulation  ###############################################
    ###########################################################################
    @staticmethod
    def module_defaults() -> typing.Union[typing.List[typing.Tuple], typing.List]:
        return [(mantrap.modules.GoalNormModule, {"optimize_speed": False, "weight": 1.0}),
                (mantrap.modules.baselines.InteractionPositionModule, {"weight": 1.0}),
                mantrap.modules.ControlLimitModule,
                mantrap.modules.SpeedLimitModule,
                mantrap.modules.baselines.MinDistanceModule]

    ###########################################################################
    # Solver properties #######################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "mcts"
