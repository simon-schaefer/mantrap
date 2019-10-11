from typing import Any, Dict

import numpy as np

from murseco.obstacle.abstract import DiscreteTimeObstacle
from murseco.utility.stats import Gaussian2D


class SingleModeDiscreteTimeObstacle(DiscreteTimeObstacle):
    """The SingleModeDiscreteTimeObstacle obstacle is a simplification of the generalized TGMM model, basically
    assuming just a single and iid Gaussian distribution in every time-step.

    :argument history: initial 2D position of obstacle (2,) or history of previous positions (n, 2).
    :argument: sigma: covariance matrices for gaussian(2, 2).
    """

    def __init__(self, history: np.ndarray = np.zeros(2), covariance: np.ndarray = np.eye(2), **kwargs):
        kwargs.update({"name": "obstacle/cardinal/SingleModeDiscreteTimeObstacle"})
        super(SingleModeDiscreteTimeObstacle, self).__init__(history, **kwargs)

        assert covariance.shape == (2, 2), "covariance matrix must be of shape (2, 2)"
        self._covariance = covariance

    def pdf(self, history: np.ndarray = None) -> Gaussian2D:
        history = super(SingleModeDiscreteTimeObstacle, self).pdf(history)
        return Gaussian2D(history[-1, :], self._covariance)

    def summary(self) -> Dict[str, Any]:
        summary = super(SingleModeDiscreteTimeObstacle, self).summary()
        summary.update({"covariance": self._covariance.tolist()})
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(SingleModeDiscreteTimeObstacle, cls).from_summary(json_text)
        covariance = np.reshape(np.array(json_text["covariance"]), (2, 2))
        summary.update({"covariance": covariance})
        return cls(**summary)
