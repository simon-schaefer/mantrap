import logging
from typing import Any, List, Dict

import numpy as np

from murseco.obstacle.abstract import DTVObstacle
from murseco.utility.stats import Gaussian2D, GMM2D


class StaticDTVObstacle(DTVObstacle):
    """The StaticDTVObstacle obstacle is static in space and time, i.e. static in position-space due to zero velocity.

    :argument mu: mean vector for position gaussian (2,) [m].
    :argument: covariance: covariance matrices for position gaussian (2, 2).
    """

    def __init__(self, mu: np.ndarray = np.zeros(2), covariance: np.ndarray = np.eye(2), **kwargs):
        kwargs.update({"name": "obstacle/static/StaticDTVObstacle", "num_modes": 1, "history": mu})
        super(StaticDTVObstacle, self).__init__(**kwargs)

        assert mu.size == 2, "mean must be two-dimensional vector"
        assert covariance.shape == (2, 2), "covariance matrix must be of shape (2, 2)"

        self._position_mu = mu
        self._position_cov = covariance

    def tppdf(self, thorizon: int, num_samples_per_mode: int = None, mproc: bool = None) -> List[GMM2D]:
        logging.debug(f"{self.name}: building static obstacle GMM")
        mus = np.reshape(self._position_mu, (1, 2))
        sigmas = np.reshape(self._position_cov, (1, 2, 2))
        return [GMM2D(mus, sigmas, weights=np.ones(1))] * thorizon

    def trajectory_samples(self, thorizon: int, num_samples: int, mode: int = None, mproc: bool = True) -> np.ndarray:
        samples = [Gaussian2D(self._position_mu, self._position_cov).sample(thorizon) for _ in range(num_samples)]
        return np.array(samples)

    def vpdf(self, history: np.ndarray = None) -> Gaussian2D:
        super(StaticDTVObstacle, self).vpdf(history)
        return Gaussian2D(np.zeros(2), np.eye(2) * 1e-8)

    def vpdf_by_mode(self, mode: int, history: np.ndarray = None) -> Gaussian2D:
        assert mode == 0, "object has only one mode"
        return self.vpdf()

    def summary(self) -> Dict[str, Any]:
        summary = super(StaticDTVObstacle, self).summary()
        summary.update({"mu": self._position_mu.tolist(), "covariance": self._position_cov.tolist()})
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(StaticDTVObstacle, cls).from_summary(json_text)
        mu = np.reshape(np.array(json_text["mu"]), (2,))
        cov = np.reshape(np.array(json_text["covariance"]), (2, 2))
        summary.update({"mu": mu, "covariance": cov})
        return cls(**summary)
