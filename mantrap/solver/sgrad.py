import typing

import mantrap.constants

from .base.ipopt import IPOPTIntermediate
from .base.z_controls import ZControlIntermediate


class SGradSolver(IPOPTIntermediate, ZControlIntermediate):
    """Shooting NLP using IPOPT solver.

    .. math:: z = controls
    .. math:: J(z) = J_{goal} + J_{interaction}
    .. math:: C(z) = [C_{max-speed}, C_{min-distance}]
    """

    ###########################################################################
    # Optimization formulation - Objective ####################################
    ###########################################################################
    @staticmethod
    def objective_defaults() -> typing.List[typing.Tuple[str, float]]:
        return [(mantrap.constants.OBJECTIVE_GOAL, 1.0),
                (mantrap.constants.OBJECTIVE_INTERACTION_POS, 10.0)]

    ###########################################################################
    # Optimization formulation - Constraints ##################################
    ###########################################################################
    @staticmethod
    def constraints_defaults() -> typing.List[str]:
        return [mantrap.constants.CONSTRAINT_CONTROL_LIMIT,
                mantrap.constants.CONSTRAINT_NORM_DISTANCE]

    ###########################################################################
    # Solver properties #######################################################
    ###########################################################################
    @staticmethod
    def solver_name() -> str:
        return "sgrad"
