import typing

import mantrap.constraints
import mantrap.objectives

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
    def objective_defaults() -> typing.List[typing.Tuple[mantrap.objectives.ObjectiveModule.__class__, float]]:
        return [(mantrap.objectives.GoalModule, 1.0), (mantrap.objectives.InteractionPositionModule, 10.0)]

    ###########################################################################
    # Optimization formulation - Constraints ##################################
    ###########################################################################
    @staticmethod
    def constraints_defaults() -> typing.List[mantrap.constraints.ConstraintModule.__class__]:
        return [mantrap.constraints.ControlLimitModule, mantrap.constraints.NormDistanceModule]

    ###########################################################################
    # Solver properties #######################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "sgrad"
