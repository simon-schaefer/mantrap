from typing import List, Tuple, Union

import numpy as np
import torch

from mantrap.constants import AGENT_SPEED_MAX
from mantrap.solver.constraints.constraint_module import ConstraintModule


class MaxSpeedModule(ConstraintModule):
    """Maximal ego speed of every trajectory point.

    For computing this constraint simply the norm of the planned velocity is determined and compared to the maximal
    agent's speed limit. For 0 < t < T_{planning}:

    .. math:: vel(t) < v_{max}

    :param horizon: planning time horizon in number of time-steps (>= 1).
    """
    def initialize(self, **module_kwargs):
        pass

    def constraint_bounds(self) -> Tuple[Union[np.ndarray, List[None]], Union[np.ndarray, List[None]]]:
        return np.ones(self.num_constraints) * (-AGENT_SPEED_MAX), np.ones(self.num_constraints) * AGENT_SPEED_MAX

    def _compute(self, ego_trajectory: torch.Tensor, ado_ids: List[str] = None) -> torch.Tensor:
        return ego_trajectory[:, 2:4].flatten()

    @property
    def num_constraints(self) -> int:
        return 2 * (self.T + 1)
