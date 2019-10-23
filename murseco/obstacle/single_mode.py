from typing import Any, Dict

import numpy as np

from murseco.obstacle.abstract import DTVObstacle
from murseco.utility.stats import Gaussian2D


class SingleModeDTVObstacle(DTVObstacle):
    """The SingleModeDTVObstacle obstacle is a simplification of the generalized TGMM model, basically
    assuming just a single and iid Gaussian distribution for the velocity prediction in every time-step.

    :argument history: initial 2D position of obstacle (2,) or history of previous positions (n, 2).
    :argument mu: mean vector for velocity gaussian (2,) [m/s].
    :argument: covariance: covariance matrices for velocity gaussian (2, 2).
    """

    def __init__(
        self,
        history: np.ndarray = np.zeros(2),
        mu: np.ndarray = np.zeros(2),
        covariance: np.ndarray = np.eye(2),
        **kwargs
    ):
        kwargs.update({"name": "obstacle/single_mode/SingleModeDTVObstacle", "num_modes": 1})
        super(SingleModeDTVObstacle, self).__init__(history, **kwargs)

        assert mu.size == 2, "mean must be two-dimensional vector"
        assert covariance.shape == (2, 2), "covariance matrix must be of shape (2, 2)"

        self._velocity_mu = mu
        self._velocity_cov = covariance

    def vpdf(self, history: np.ndarray = None) -> Gaussian2D:
        super(SingleModeDTVObstacle, self).vpdf(history)
        return Gaussian2D(self._velocity_mu, self._velocity_cov)

    def vpdf_by_mode(self, mode: int, history: np.ndarray = None) -> Gaussian2D:
        assert mode == 0, "object has only one mode"
        return self.vpdf()

    def summary(self) -> Dict[str, Any]:
        summary = super(SingleModeDTVObstacle, self).summary()
        summary.update({"mu": self._velocity_mu.tolist(), "covariance": self._velocity_cov.tolist()})
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(SingleModeDTVObstacle, cls).from_summary(json_text)
        mu = np.reshape(np.array(json_text["mu"]), (2,))
        covariance = np.reshape(np.array(json_text["covariance"]), (2, 2))
        summary.update({"mu": mu, "covariance": covariance})
        return cls(**summary)
