import numpy as np
import torch

import mantrap.constants

from .attention_module import AttentionModule


class ClosestModule(AttentionModule):
    """Attention based for closest, limited-range pedestrian only.

    The closest filter selects the ado that is closest to the robot and thereby is closer than a certain
    euclidean distance. Thereby merely the positions at time t = t_current are taken into account

    .. math:: ||pos_{ego}(t) - pos_{ado}(t)||_2 < R_{attention}

    R_{attention} is called attention radius and is the maximal L2-distance for an ado from the ego to be taken into
    account for planning (nevertheless it will always be taken into account for forward simulations, in order to
    prevent deviating much from the actual full-agent planning due to possible behavioral changes of the ados with
    less agents in the scene).
    """
    def _compute(self) -> np.ndarray:
        with torch.no_grad():
            ego_state, ado_states = self._env.states()
            euclidean_distances = torch.norm(ado_states[:, 0:2] - ego_state[0:2], dim=1)
            index_min = torch.argmin(euclidean_distances)
            euclidean_distances[~index_min] = np.inf  # all but the closest index
            in_attention = (euclidean_distances < mantrap.constants.ATTENTION_CLOSEST_RADIUS).numpy()
            in_indices = np.nonzero(in_attention)[0]

        return in_indices

    ###########################################################################
    # Filter properties #######################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "closest"
