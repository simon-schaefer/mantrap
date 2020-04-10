import numpy as np
import torch

from mantrap.solver.filter.filter_module import FilterModule
from mantrap.utility.shaping import check_ado_states


class NoFilterModule(FilterModule):
    """No filter i.e. include all agents for planning.

    This module is created for easier treating of the exception of not applying a filter (filter = None).
    """

    def _compute(self, ego_state: torch.Tensor, ado_states: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            assert check_ado_states(ado_states, enforce_temporal=False)
            num_ados, _ = ado_states.shape
        return np.arange(start=0, stop=num_ados, step=1)
