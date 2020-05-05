import numpy as np

from .filter_module import FilterModule


class NoFilterModule(FilterModule):
    """No filter i.e. include all agents for planning.

    This module is created for easier treating of the exception of not applying a filter (filter = None).
    """
    def _compute(self) -> np.ndarray:
        return np.arange(start=0, stop=self._env.num_ados, step=1)
