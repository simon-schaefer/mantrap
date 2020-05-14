import typing

import numpy as np

import mantrap.modules

from ..base import SearchIntermediate, ZControlIntermediate


class RandomSearch(ZControlIntermediate, SearchIntermediate):

    def _optimize_inner(self, z_best: np.ndarray, obj_best: float, iteration: int, tag: str, ado_ids: typing.List[str]
                        ) -> typing.Tuple[np.ndarray, float, int, bool]:
        """Inner optimization/search function.

        ATTENTION: See self-containment comment in `_optimize()` method description.

        In random search in every step we basically use sample a new assignment of z within its bounds and
        compare it to the best assignment so far. If it is feasible and has a smaller objective value,
        update the best assignment and objective. Random search should exploit the full allowed runtime,
        therefore never terminate (termination flag = False).

        :param z_best: best assignment of optimization vector so far.
        :param obj_best: according objective function value.
        :param iteration: current search (outer loop) iteration.
        :param tag: name of optimization call (name of the core).
        :param ado_ids: identifiers of ados that should be taken into account during optimization.
        :returns: updated best z-values, updated best objective, outer loop iteration, termination flag.
        """

        z_sample = np.random.uniform(*self.z_bounds)
        objective, constraint_violation = self._evaluate(z=z_sample, tag=tag, ado_ids=ado_ids)

        if objective < obj_best  and constraint_violation < mantrap.constants.SOLVER_CONSTRAINT_LIMIT:
            obj_best = objective
            z_best = z_sample

        return z_best, obj_best, iteration + 1, False

    ###########################################################################
    # Optimization formulation  ###############################################
    ###########################################################################
    def module_defaults(self) -> typing.List[typing.Tuple]:
        return [(mantrap.modules.GoalModule, {"optimize_speed": False, "weight": 1.0}),
                (mantrap.modules.InteractionPositionModule, {"weight": 1.0}),
                (mantrap.modules.ControlLimitModule, None),
                (mantrap.modules.MinDistanceModule, None)]

    ###########################################################################
    # Solver properties #######################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "random_search"
