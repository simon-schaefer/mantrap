from typing import Any, Dict, List

import numpy as np

from murseco.obstacle.abstract import DiscreteTimeObstacle
from murseco.utility.stats import GMM2D


class TGMMDiscreteTimeObstacle(DiscreteTimeObstacle):
    """The GMMObstacle class represents an obstacle as 2D Gaussian Mixture Model that is evolving over time.
    The parameters of the model are thereby stated as stacked arrays with the first dimension describing the
    model's time-step, the remaining dimensions describing the model itself.

    :argument tmus: mean vectors over time (time-step, mode, 2).
    :argument tsigmas: covariance matrices over time (time-step, mode, 2, 2).
    :argument tweights: weight vector over time (time-step, mode).
    """

    def __init__(self, tmus: np.ndarray, tsigmas: np.ndarray, tweights: np.ndarray):
        super(TGMMDiscreteTimeObstacle, self).__init__("obstacle/tgmm/TGMMDiscreteTimeObstacle", tmus.shape[0])
        assert len(tmus.shape) == 3, "tmus must be in shape (time-step, mode, 2)"
        assert len(tsigmas.shape) == 4, "tsigmas must be in shape (time-step, mode, 2, 2)"
        assert len(tweights.shape) == 2, "tweights must be in shape (time-step, mode)"
        assert tmus.shape[0] == tsigmas.shape[0] == tweights.shape[0], "time dimensions must be equal"

        self._tgmm = [GMM2D(tmus[t, :, :], tsigmas[t, :, :, :], tweights[t, :]) for t in range(self.tmax)]

    @property
    def pdf(self) -> GMM2D:
        return self._tgmm[0]

    @property
    def tpdf(self) -> List[GMM2D]:
        return self._tgmm

    def summary(self) -> Dict[str, Any]:
        summary = super(TGMMDiscreteTimeObstacle, self).summary()
        summary.update({"tgmms": [g.summary() for g in self._tgmm]})
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        super(TGMMDiscreteTimeObstacle, cls).from_summary(json_text)
        gmms = [GMM2D.from_summary(gmm_text) for gmm_text in json_text["tgmms"]]
        num_time_steps = len(gmms)
        tmus = np.reshape(np.array([[g.mu for g in gmm.gaussians] for gmm in gmms]), (num_time_steps, -1, 2))
        tsigmas = np.reshape(np.array([[g.sigma for g in gmm.gaussians] for gmm in gmms]), (num_time_steps, -1, 2, 2))
        tweights = np.reshape(np.array([gmm.weights for gmm in gmms]), (num_time_steps, -1))
        return cls(tmus, tsigmas, tweights)
