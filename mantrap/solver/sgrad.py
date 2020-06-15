import typing

import mantrap.modules

from .base import IPOPTIntermediate


class SGradSolver(IPOPTIntermediate):
    """Shooting NLP using IPOPT solver.

    .. math:: z = controls
    .. math:: J(z) = J_{goal} + J_{interaction}
    .. math:: C(z) = [C_{max-speed}, C_{min-distance}]
    """

    ###########################################################################
    # Optimization formulation  ###############################################
    ###########################################################################
    @staticmethod
    def module_defaults() -> typing.Union[typing.List[typing.Tuple], typing.List]:
        return [(mantrap.modules.GoalNormModule, {"optimize_speed": False, "weight": 1.0}),
                (mantrap.modules.InteractionProbabilityModule, {"weight": 1.0}),
                mantrap.modules.SpeedLimitModule,
                mantrap.modules.HJReachabilityModule]

    ###########################################################################
    # Solver properties #######################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "sgrad"
