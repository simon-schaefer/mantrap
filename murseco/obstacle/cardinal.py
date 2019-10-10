from typing import Any, Dict, List, Union

import numpy as np

from murseco.obstacle.abstract import DiscreteTimeObstacle
from murseco.utility.misc import cardinal_directions
from murseco.utility.stats import Distribution2D, Point2D, GMM2D


class CardinalDiscreteTimeObstacle(DiscreteTimeObstacle):
    """The CardinalDiscreteTime obstacle is a simplification of the generalized TGMM obstacle. The cardinal obstacle
    assumes that the obstacle can merely move into cardinal directions (up, down, left, right) without necessarily
    being in a grid world (!). Thus, tn = t0 is a cardinal 4-modal GMM distribution, in each step tn = 1 ... tmax
    a direction is drawn from the previous distribution, determining the new position. Following the second assumption
    the new GMM distribution is similar to the previous one, just with updated mean values (delta_position added).

    :argument pinit: initial 2D position of obstacle (2,).
    :argument pstep: step distance to go in every time-step either as float (uniform for all directions) or 4d-array.
    :argument tmax: maximal number of time-steps.
    :argument: sigmas: covariance matrices for every cardinal direction (4, 2, 2).
    :argument weights: weights of distribution in every direction (4,).
    """

    def __init__(
        self,
        pinit: np.ndarray,
        pstep: Union[float, np.ndarray],
        tmax: int,
        sigmas: np.ndarray,
        weights: np.ndarray,
        tgmm: List[Distribution2D] = None,
    ):
        super(CardinalDiscreteTimeObstacle, self).__init__("obstacle/cardinal/CardinalDiscreteTimeObstacle", tmax)
        pstep = np.ones(4) * pstep if type(pstep) != np.ndarray else pstep

        assert pinit.size == 2, "initial position must be two-dimensional"
        assert pstep.size == 4, "pstep must contain the step distance in all cardinal directions"
        assert sigmas.shape == (4, 2, 2), "sigmas matrix must be of shape (4, 2, 2) containing 4 covariance matrices "

        self._position = pinit
        self._pstep = np.reshape(pstep, (4, 1))
        self._sigmas, self._weights = sigmas, weights
        self._tgmm = tgmm if tgmm is not None else self._plan_tgmm()

    @property
    def position(self) -> np.ndarray:
        return self._position

    @property
    def pdf(self) -> Distribution2D:
        return self._tgmm[0]

    @property
    def tpdf(self) -> List[Distribution2D]:
        return self._tgmm

    def _plan_tgmm(self) -> List[Distribution2D]:
        # Initial distribution (tn = 0) is DirecDelta distribution at the current position.
        dist_init = Point2D(self.position)
        tgmm = [dist_init]

        # Create initial distribution (i.e. GMM(tn = 1)) from position, step and sigmas.
        mus = self.position + cardinal_directions() * self._pstep
        tgmm.append(GMM2D(mus, self._sigmas, self._weights))

        # For whole time range (tn = 2...tmax), repeat:
        # 1) Sample position to go from GMM(t-1)
        # 2) Assign sampled position to position(t)
        # 3) Build GMM(t) by updating mu values
        # 4) Repeat until tn = tmax
        for tn in range(2, self.tmax):
            position_next = tgmm[-1].sample(num_samples=1)
            mus = position_next + cardinal_directions() * self._pstep
            tgmm.append(GMM2D(mus, self._sigmas, self._weights))
        return tgmm

    def summary(self) -> Dict[str, Any]:
        summary = super(CardinalDiscreteTimeObstacle, self).summary()
        summary.update(
            {
                "position": self.position.tolist(),
                "pstep": self._pstep.tolist(),
                "sigmas": self._sigmas.tolist(),
                "weights": self._weights.tolist(),
                "tgmms": [g.summary() for g in self._tgmm],
            }
        )
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        super(CardinalDiscreteTimeObstacle, cls).from_summary(json_text)
        tgmm = [cls.call_by_summary(gmm_text) for gmm_text in json_text["tgmms"]]
        position = np.reshape(np.array(json_text["position"]), (2,))
        pstep = np.reshape(np.array(json_text["pstep"]), (4, 1))
        sigmas = np.reshape(np.array(json_text["sigmas"]), (4, 2, 2))
        weights = np.reshape(np.array(json_text["weights"]), (4,))
        return cls(position, pstep, json_text["tmax"], sigmas, weights, tgmm)
