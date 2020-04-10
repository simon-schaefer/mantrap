from typing import Tuple

import numpy as np
import torch

from mantrap.constants import filter_euclidean_radius
from mantrap.solver.filter.filter_module import FilterModule
from mantrap.utility.shaping import check_ego_state, check_ado_states


class EuclideanModule(FilterModule):
    """Filter based on the euclidean distance between current ego's and ado's positions.

    The euclidean filter selects the ados that are close, i.e. in a certain euclidean distance from the ego position.
    Thereby merely the positions at time t = t_current are taken into account

    .. math:: ||pos_{ego}(t) - pos_{ado}(t)||_2 < R_{attention}

    R_{attention} is called attention radius and is the maximal L2-distance for an ado from the ego to be taken into
    account for planning (nevertheless it will always be taken into account for forward simulations, in order to
    prevent deviating much from the actual full-agent planning due to possible behavioral changes of the ados with
    less agents in the scene).
    """

    def _compute(self, scene_states: Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        with torch.no_grad():
            ego_state, ados_states = scene_states
            assert check_ego_state(ego_state, enforce_temporal=False)
            assert check_ado_states(ados_states, enforce_temporal=False)

            euclidean_distances = torch.norm(ados_states[:, 0:2] - ego_state[0:2], dim=1)
            in_attention = (euclidean_distances < filter_euclidean_radius).numpy()
            in_indices = np.nonzero(in_attention)[0]

        return in_indices
