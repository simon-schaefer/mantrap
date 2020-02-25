from typing import Tuple

import numpy as np
import torch

from mantrap.constants import agent_speed_max
from mantrap.solver.constraints.constraint_module import ConstraintModule


class MaxSpeedModule(ConstraintModule):

    def __init__(self, **module_kwargs):
        super(MaxSpeedModule, self).__init__(**module_kwargs)

    def constraint_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.ones(self.T * 2) * (-agent_speed_max), np.ones(self.T * 2) * agent_speed_max

    def _compute(self, x4: torch.Tensor) -> torch.Tensor:
        return x4[:, 2:4].flatten()
