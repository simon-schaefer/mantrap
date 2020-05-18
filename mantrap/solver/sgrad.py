import typing

import mantrap.modules

from .base import IPOPTIntermediate, ZControlIntermediate


class SGradSolver(IPOPTIntermediate, ZControlIntermediate):
    """Shooting NLP using IPOPT solver.

    .. math:: z = controls
    .. math:: J(z) = J_{goal} + J_{interaction}
    .. math:: C(z) = [C_{max-speed}, C_{min-distance}]
    """

    ###########################################################################
    # Optimization formulation  ###############################################
    ###########################################################################
    def module_defaults(self) -> typing.List[typing.Tuple]:
        return [(mantrap.modules.GoalNormModule, {"optimize_speed": False, "weight": 1.0}),
                (mantrap.modules.InteractionPositionModule, {"weight": 1.0}),
                (mantrap.modules.ControlLimitModule, None),
                (mantrap.modules.MinDistanceModule, None)]

    ###########################################################################
    # Solver properties #######################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "sgrad"
