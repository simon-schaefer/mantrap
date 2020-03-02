from typing import List, Tuple, Union

import numpy as np
import torch

from mantrap.constants import agent_speed_max
from mantrap.solver.constraints.constraint_module import ConstraintModule


class MaxSpeedModule(ConstraintModule):

    def initialize(self, **module_kwargs):
        pass

    def constraint_bounds(self) -> Tuple[Union[np.ndarray, List[None]], Union[np.ndarray, List[None]]]:
        return np.ones(self.num_constraints) * (-agent_speed_max), np.ones(self.num_constraints) * agent_speed_max

    def _compute(self, x4: torch.Tensor) -> torch.Tensor:
        return x4[:, 2:4].flatten()

    @property
    def num_constraints(self) -> int:
        return 2 * (self.T + 1)
