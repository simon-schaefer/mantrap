from typing import Any, Dict

import numpy as np

from murseco.obstacle.abstract import DTVObstacle
from murseco.utility.stats import Gaussian2D, GMM2D


class AngularDTVObstacle(DTVObstacle):
    """The AngularDTVObstacle obstacle is a simplification of the generalized TGMM model, basically
    assuming just a multimodal but iid Gaussian distribution for the velocity prediction in every time-step.
    However when the adapting flag is set the weights of different modes change based on the past direction of
    movement, so basically once the object starts moving in some direction it is more likely to continue moving
    in this direction (by increasing the weight of that mode).

    :argument history: initial 2D position of obstacle (2,) or history of previous positions (n, 2).
    :argument mus: mean vector for velocity gaussian (2,) [m/s] for all modes (num_modes, 2).
    :argument: covariances: covariance matrices for velocity gaussian (2, 2) for all modes (num_modes, 2, 2).
    :argument weights: weights of distribution in every direction (num_modes, 4).
    :argument is_adapting: weights adapt with every history (resolves iid assumption !!! but more realistic)
    """

    def __init__(
        self,
        history: np.ndarray = np.zeros(2),
        mus: np.ndarray = np.array([[-1, 1], [0, 1], [1, 1]]),
        covariances: np.ndarray = np.array([np.eye(2) * 0.01] * 3),
        weights: np.ndarray = np.ones(3),
        is_adapting: bool = True,
        num_modes: int = None,
        **kwargs,
    ):
        kwargs.update({"name": "obstacle/angular/AngularDTVObstacle"})
        num_modes = mus.size // 2 if num_modes is None else num_modes
        super(AngularDTVObstacle, self).__init__(history, num_modes=num_modes, **kwargs)

        assert mus.size == self.num_modes * 2, "mean vector must be two-dimensional for every mode"
        assert covariances.size == self.num_modes * 2 * 2, f"covariances must be of shape ({self.num_modes}, 2, 2)"
        assert all(weights >= 0), "weights must be semi-positive"
        assert weights.size == self.num_modes, "number of weights must be equal to number of modes"

        self._mus = np.reshape(mus, (self.num_modes, 2))
        self._covariances = np.reshape(covariances, (self.num_modes, 2, 2))
        self._weights = weights
        self._is_adapting = is_adapting

    def vpdf(self, history: np.ndarray = None) -> GMM2D:
        history = super(AngularDTVObstacle, self).vpdf(history)
        if not self._is_adapting or history.shape[0] < 2:
            weights = self._weights
        # If the weights are chosen to adapt and the history is long enough, the weights are re-distributed by
        # weighting the mode vectors using the last direction the object has moved to. Mathematically speaking
        # the difference in position is transformed to the ND "mode vector" space, the positive coordinates
        # (only positives because otherwise ambiguous map) build the weight of the last movement direction.
        # Example for the cardinal vector space i.e. if the means would be the 4 cardinal directions:
        # dp = [0.5, -1] ==> w = (T * dp)_+ = ([0.5 -0.5 -0.5 1])_+ = [0.5 0 0 1]
        else:
            T = self._mus.copy()
            dp = history[-1, :] - history[-2, :]
            weights_history = np.dot(T, dp)
            weights = self._weights / np.sum(self._weights) + 20 * np.abs(weights_history / np.sum(weights_history))
            weights = weights / np.sum(weights)
        return GMM2D(self._mus, self._covariances, weights)

    def vpdf_by_mode(self, mode: int, history: np.ndarray = None) -> Gaussian2D:
        assert 0 <= mode <= self._num_modes - 1, f"object only has {self._num_modes} modes (not mode {mode}"
        return self.vpdf(history).mode(mode)

    def summary(self) -> Dict[str, Any]:
        summary = super(AngularDTVObstacle, self).summary()
        summary.update(
            {
                "mus": self._mus.tolist(),
                "covariances": self._covariances.tolist(),
                "weights": self._weights.tolist(),
                "is_adapting": self._is_adapting,
            }
        )
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(AngularDTVObstacle, cls).from_summary(json_text)
        num_modes = summary["num_modes"]
        mus = np.reshape(np.array(json_text["mus"]), (num_modes, 2))
        covariances = np.reshape(np.array(json_text["covariances"]), (num_modes, 2, 2))
        weights = np.reshape(np.array(json_text["weights"]), (num_modes,))
        is_adapting = str(json_text["is_adapting"]) == "True"
        summary.update({"mus": mus, "covariances": covariances, "weights": weights, "is_adapting": is_adapting})
        return cls(**summary)
