from typing import Any, Dict, Union

import numpy as np

from murseco.obstacle.abstract import DiscreteTimeObstacle
from murseco.utility.misc import cardinal_directions
from murseco.utility.stats import Distribution2D, GMM2D


class CardinalDiscreteTimeObstacle(DiscreteTimeObstacle):
    """The CardinalDiscreteTime obstacle is a simplification of the generalized TGMM obstacle. The cardinal obstacle
    assumes that the obstacle can merely move into cardinal directions (up, down, left, right) without necessarily
    being in a grid world (!). Thus, tn = t0 is a cardinal 4-modal GMM distribution, having different weights and
    velocities (step-distance) for every direction. A strong simplification made within this obstacle type definition
    is that the probability distribution of the next position does not (!) depend on the history of previous position,
    so it is always the same GMM iid to the previous ones.

    :argument history: initial 2D position of obstacle (2,) or history of previous positions (n, 2).
    :argument velocity: step distance to go in every time-step either as float (uniform for all directions) or 4d-array.
    :argument: sigmas: covariance matrices for every cardinal direction (4, 2, 2).
    :argument weights: weights of distribution in every direction (4,).
    """

    def __init__(
        self,
        history: np.ndarray = np.zeros(2),
        velocity: Union[float, np.ndarray] = 1.0,
        sigmas: np.ndarray = np.array([np.eye(2)] * 4),
        weights: np.ndarray = np.ones(4),
        **kwargs
    ):
        kwargs.update({"name": "obstacle/cardinal/CardinalDiscreteTimeObstacle"})
        super(CardinalDiscreteTimeObstacle, self).__init__(history, **kwargs)
        velocity = np.ones(4) * velocity if type(velocity) != np.ndarray else velocity

        assert velocity.size == 4, "pstep must contain the step distance in all cardinal directions"
        assert sigmas.shape == (4, 2, 2), "sigmas matrix must be of shape (4, 2, 2) containing 4 covariance matrices"

        self._velocity = np.reshape(velocity, (4, 1))
        self._sigmas, self._weights = sigmas, weights

    def pdf(self, history: np.ndarray = None) -> Distribution2D:
        history = super(CardinalDiscreteTimeObstacle, self).pdf(history)
        position = history[-1, :]
        mus = position + cardinal_directions() * self._velocity
        return GMM2D(mus, self._sigmas, self._weights)

    def summary(self) -> Dict[str, Any]:
        summary = super(CardinalDiscreteTimeObstacle, self).summary()
        summary.update(
            {"velocity": self._velocity.tolist(), "sigmas": self._sigmas.tolist(), "weights": self._weights.tolist()}
        )
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(CardinalDiscreteTimeObstacle, cls).from_summary(json_text)
        velocity = np.reshape(np.array(json_text["velocity"]), (4, 1))
        sigmas = np.reshape(np.array(json_text["sigmas"]), (4, 2, 2))
        weights = np.reshape(np.array(json_text["weights"]), (4,))
        summary.update({"velocity": velocity, "sigmas": sigmas, "weights": weights})
        return cls(**summary)
