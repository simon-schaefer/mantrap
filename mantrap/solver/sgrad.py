from typing import List, Tuple

from mantrap.constants import *
from mantrap.solver.solver_intermediates.ipopt import IPOPTIntermediate
from mantrap.solver.solver_intermediates.z_controls import ZControlIntermediate


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
    def objective_defaults() -> List[Tuple[str, float]]:
        return [(OBJECTIVE_GOAL, 1.0), (OBJECTIVE_INTERACTION_POS, 10.0)]

    ###########################################################################
    # Optimization formulation - Constraints ##################################
    ###########################################################################
    @staticmethod
    def constraints_defaults() -> List[str]:
        return [CONSTRAINT_CONTROL_LIMIT, CONSTRAINT_NORM_DISTANCE]

    ###########################################################################
    # Solver properties #######################################################
    ###########################################################################
    @staticmethod
    def solver_name() -> str:
        return "sgrad"
