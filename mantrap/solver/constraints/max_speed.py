from typing import List, Tuple, Union

import numpy as np
import torch

from mantrap.constants import AGENT_SPEED_MAX
from mantrap.solver.constraints.constraint_module import ConstraintModule


class MaxSpeedModule(ConstraintModule):
    """Maximal ego speed of every trajectory point.

    For computing this constraint simply the norm of the planned velocity is determined and compared to the maximal
    agent's speed limit. For 0 < t < T_{planning}:

    .. math:: speed(t) < v_{max}

    :param horizon: planning time horizon in number of time-steps (>= 1).
    """
    def initialize(self, **module_kwargs):
        pass

    def _compute(self, ego_trajectory: torch.Tensor, ado_ids: List[str] = None) -> torch.Tensor:
        """Determine constraint value core method.

        The max speed constraints simply are the velocity values of the ego trajectory.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        return torch.norm(ego_trajectory[:, 2:4], dim=1).flatten()

    def _constraints_gradient_condition(self) -> bool:
        """Conditions for the existence of a gradient between the input of the constraint value computation
        (which is the ego_trajectory) and the constraint values itself. If returns True and the ego_trajectory
        itself requires a gradient, the constraint output has to require a gradient as well.

        Since the velocities are part of the given ego_trajectory, the gradient should always exist.
        """
        return True

    @property
    def constraint_bounds(self) -> Tuple[Union[float, None], Union[float, None]]:
        """Lower and upper bounds for constraint values.

        For the max speed constraint the lower is None since the norm always is semi-positive and upper bounds
        are the agents maximal allowed speeds, which is an assumed constant value defined in constants.
        """
        return None, AGENT_SPEED_MAX

    @property
    def num_constraints(self) -> int:
        return self.T + 1
