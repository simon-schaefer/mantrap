from typing import Tuple

import numpy as np
import torch

from mantrap.solver.filter.filter_module import FilterModule
from mantrap.utility.shaping import check_ado_states


class NoFilterModule(FilterModule):
    """No filter i.e. include all agents for planning.

    This module is created for easier treating of the exception of not applying a filter (filter = None).
    """

    def _compute(self, scene_states: Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        with torch.no_grad():
            _, ados_states = scene_states
            assert check_ado_states(ados_states, enforce_temporal=False)
            num_ados, _ = ados_states.shape
        return np.arange(start=0, stop=num_ados, step=1)
