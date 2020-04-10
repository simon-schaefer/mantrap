from abc import ABC, abstractmethod
import logging

import numpy as np
import torch

from mantrap.utility.shaping import check_ego_state, check_ado_states


class FilterModule(ABC):
    """General filter class.

    The filter selects the "important" ados/modes from the list of all ados in the scene. The selection is returned
    as list of indices of the chosen ados which should be taken into account in further computations.
    For a unified  and general implementation of the filter modules this superclass implements methods for computing
    and logging the filter, all based on the `_compute()` method which should be implemented in the child classes.
    """
    def __init__(self, **module_kwargs):
        pass

    ###########################################################################
    # Filter Formulation ######################################################
    ###########################################################################
    def compute(self, ego_state: torch.Tensor, ado_states: torch.Tensor) -> np.ndarray:
        assert check_ego_state(x=ego_state, enforce_temporal=False)
        assert check_ado_states(x=ado_states, enforce_temporal=False)

        filtered_indices = self._compute(ego_state=ego_state, ado_states=ado_states)
        return self._return_filtered(filtered_indices)

    @abstractmethod
    def _compute(self, ego_state: torch.Tensor, ado_states: torch.Tensor) -> np.ndarray:
        raise NotImplementedError

    ###########################################################################
    # Utility #################################################################
    ###########################################################################
    def _return_filtered(self, indices_filtered: np.ndarray) -> np.ndarray:
        self._indices_current = indices_filtered
        logging.debug(f"Module {self.__str__()} computed")
        return self._indices_current

    ###########################################################################
    # Filter properties #######################################################
    ###########################################################################
    @property
    def indices_current(self) -> np.ndarray:
        return self._indices_current
