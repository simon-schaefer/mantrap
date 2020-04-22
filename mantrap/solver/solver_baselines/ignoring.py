from typing import List, Tuple

import torch

from mantrap.constants import *
from mantrap.solver.solver_intermediates.ipopt import IPOPTIntermediate
from mantrap.solver.solver_intermediates.z_controls import ZControlIntermediate


class IgnoringSolver(IPOPTIntermediate, ZControlIntermediate):
    """Shooting NLP using IPOPT solver.

    .. math:: z = controls
    .. math:: J(z) = J_{goal}
    .. math:: C(z) = [C_{max-speed}]
    """

    ###########################################################################
    # Initialization ##########################################################
    ###########################################################################
    def z0s_default(self, just_one: bool = False) -> torch.Tensor:
        z0s = super(IgnoringSolver, self).z0s_default(just_one=just_one)
        return z0s[0:1, :, :] if not just_one else z0s

    ###########################################################################
    # Optimization formulation - Objective ####################################
    ###########################################################################
    @staticmethod
    def objective_defaults() -> List[Tuple[str, float]]:
        return [(OBJECTIVE_GOAL, 1.0)]

    ###########################################################################
    # Optimization formulation - Constraints ##################################
    ###########################################################################
    @staticmethod
    def constraints_defaults() -> List[str]:
        return [CONSTRAINT_MAX_SPEED]

    ###########################################################################
    # Solver properties #######################################################
    ###########################################################################
    @property
    def solver_name(self) -> str:
        return "ignoring"
