import typing

import torch

import mantrap.constraints
import mantrap.objectives

from ..base.ipopt import IPOPTIntermediate
from ..base.z_controls import ZControlIntermediate


class IgnoringSolver(IPOPTIntermediate, ZControlIntermediate):
    """Shooting NLP using IPOPT solver.

    .. math:: z = controls
    .. math:: J(z) = J_{goal}
    .. math:: C(z) = [C_{max-speed}]
    """

    ###########################################################################
    # Initialization ##########################################################
    ###########################################################################
    def initial_values(self, just_one: bool = False) -> torch.Tensor:
        """For this solver only the goal distance is the objective to be minimized. Going there is
        fast as possible is following a straight line from start to goal, with maximal velocity
        which happens to be the second entry of the superclasses `z0s_default()` methods return.
        Having multiple starting positions does not make sense, since we know that this is the
        given the solvers objective, going straight is optimal !
        """
        z0s = super(IgnoringSolver, self).initial_values(just_one=just_one)
        return z0s[1:2, :] if not just_one else z0s

    @staticmethod
    def num_initial_values() -> int:
        return 1

    ###########################################################################
    # Optimization formulation - Objective ####################################
    ###########################################################################
    @staticmethod
    def objective_defaults() -> typing.List[typing.Tuple[mantrap.objectives.ObjectiveModule.__class__, float]]:
        return [(mantrap.objectives.GoalModule, 1.0)]

    ###########################################################################
    # Optimization formulation - Constraints ##################################
    ###########################################################################
    @staticmethod
    def constraints_defaults() -> typing.List[mantrap.constraints.ConstraintModule.__class__]:
        return [mantrap.constraints.ControlLimitModule]

    ###########################################################################
    # Solver properties #######################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "ignoring"