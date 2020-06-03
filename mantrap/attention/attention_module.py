import abc
import typing

import numpy as np

import mantrap.environment


class AttentionModule(abc.ABC):
    """General attention class.

    The filter selects the "important" ados/modes from the list of all ados in the scene. The selection is returned
    as list of indices of the chosen ados which should be taken into account in further computations.
    For a unified  and general implementation of the filter modules this superclass implements methods for computing
    and logging the filter, all based on the `_compute()` method which should be implemented in the child classes.

    :param t_horizon: planning time horizon in number of time-steps (>= 1).
    :param env: environment object reference.
    """
    def __init__(self, env: mantrap.environment.base.GraphBasedEnvironment, t_horizon: int, **unused):
        self._env = env
        self._t_horizon = t_horizon

    ###########################################################################
    # Filter Formulation ######################################################
    ###########################################################################
    def compute(self) -> typing.List[str]:
        filter_indices = self._compute()
        filtered_ids = [self._env.ado_ids[m] for m in filter_indices]
        return self._return_filtered(filtered_ids)

    @abc.abstractmethod
    def _compute(self) -> np.ndarray:
        raise NotImplementedError

    ###########################################################################
    # Utility #################################################################
    ###########################################################################
    def _return_filtered(self, ids_filtered: typing.List[str]) -> typing.List[str]:
        self._ids_current = ids_filtered
        return self._ids_current

    ###########################################################################
    # Filter properties #######################################################
    ###########################################################################
    @property
    def ids_current(self) -> typing.List[str]:
        return self._ids_current

    @property
    def env(self) -> mantrap.environment.base.GraphBasedEnvironment:
        return self._env

    @property
    def t_horizon(self) -> int:
        return self._t_horizon

    @property
    def name(self) -> str:
        raise NotImplementedError
