from abc import abstractmethod
import logging
from functools import partial
import multiprocessing
from typing import Any, Dict, List, Union

import numpy as np

from murseco.utility.io import JSONSerializer
from murseco.utility.stats import Distribution2D, GMM2D


class DTVObstacle(JSONSerializer):
    """The DiscreteTimeVelocity Obstacle is a general definition of a dynamic obstacle that is moving over time given
    some defined probability density function in the velocity space (vpdf). It implements methods to sample
    trajectories, build the next pdf in the position space and stores the objects history (position space).
    """

    def __init__(self, history: np.ndarray, num_modes: int, dt: float = 1.0, **kwargs):
        super(DTVObstacle, self).__init__(**kwargs)
        assert num_modes > 0, "number of modes must be larger than 0"
        assert history.size % 2 == 0, "history array must consist of two-dimensional points"
        assert history.size > 0, "history array must not be empty"
        assert dt > 0, "time-delta between discrete time-steps must be larger 0"

        self._history = np.reshape(history, (-1, 2))  # obstacles last position (or initial position)
        self._dt = dt
        self._num_modes = num_modes  # number of modes of obstacle distribution

    @abstractmethod
    def vpdf(self, history: np.ndarray = None) -> Union[Distribution2D, np.ndarray]:
        """Given the obstacles history (trajectory) determine the overall pdf of the next velocity."""
        return self._history.copy() if history is None else history

    @abstractmethod
    def vpdf_by_mode(self, mode: int, history: np.ndarray = None) -> Union[Distribution2D, np.ndarray]:
        """Given the obstacles history (trajectory) determine the pdf of the next velocity by mode."""
        return self._history.copy() if history is None else history

    def tppdf(self, thorizon: int, num_samples_per_mode: int = 50, mproc: bool = True) -> List[GMM2D]:
        """Build position probability density function by sampling trajectories for each mode and summarize
        them with a single gaussian distribution for each mode and time-step, given the objects history.
        Assumption: Covariance is estimated as diagonal matrix, i.e. sigma_xy = 0, for computational reasons.

        :argument thorizon: length of trajectory samples, i.e. number of predicted time-steps.
        :argument num_samples_per_mode: number of trajectory samples per mode.
        :argument mproc: run in multiprocessing (8 processes).
        :returns array of future position pdf as GMM2D for each time-step.
        """
        logging.debug(f"{self.name}: sampling trajectories")
        mus, covariances = np.zeros((self.num_modes, thorizon, 2)), np.zeros((self.num_modes, thorizon, 2))
        for m in range(self.num_modes):
            trajectories_mode = self.trajectory_samples(thorizon, num_samples_per_mode, mode=m, mproc=mproc)
            mus[m, :, :] = np.mean(trajectories_mode, axis=0)
            covariances[m, :, :] = np.std(trajectories_mode, axis=0)

        logging.debug(f"{self.name}: build gmm2d models")
        ppdf = []
        for t in range(thorizon):
            gmm_mus = np.reshape(mus[:, t, :], (self.num_modes, 2))
            # make covariance matrix invertible if its purely zeros
            gmm_cov = np.array([np.diag(covariances[m, t, :]) + np.eye(2) * 1e-6 for m in range(self.num_modes)])
            weights = np.ones(self.num_modes)
            ppdf.append(GMM2D(gmm_mus, gmm_cov, weights))
        logging.debug(f"{self.name}: ppdf done")
        return ppdf

    def trajectory_samples(self, thorizon: int, num_samples: int, mode: int = None, mproc: bool = True) -> np.ndarray:
        """Sample possible (future) trajectories given the currently stored, thorizon steps in the future.
         Therefore iteratively call the pdf() method, sample the next velocity from that pdf, forward integrate
         the samples velocity to obtain the next position, append the history and repeat until the time horizon
         (thorizon) is reached, for each trajectory sample. Each trajectory starts with the last point of the
         objects history.

         :argument thorizon: length of trajectory samples, i.e. number of predicted time-steps.
         :argument num_samples: number of trajectory samples.
         :argument mode: mode to sample trajectory from (None = overall distribution).
         :argument mproc: run in multiprocessing (8 processes).
         :returns array of sampled future trajectories (num_samples, thorizon, 2).
         """
        if mproc:
            pool = multiprocessing.Pool(8)
            func = partial(self._sample_trajectory, thorizon=thorizon, mode=mode, history=self._history)
            trajectories = pool.map(func, list(range(num_samples)))
        else:
            trajectories = []
            for _ in range(num_samples):
                trajectories.append(self._sample_trajectory(None, thorizon=thorizon, history=self._history, mode=mode))
        return np.array(trajectories)

    def _sample_trajectory(self, iteration: Union[int, None], thorizon: int, history: np.ndarray, mode: int):
        trajectory = np.zeros((thorizon, 2))
        trajectory[0, :] = history[-1, :]
        for t in range(1, thorizon):
            if mode is None:
                velocity = self.vpdf(history).sample(1)
            else:
                velocity = self.vpdf_by_mode(mode, history).sample(1)
            trajectory[t, :] = trajectory[t - 1, :] + velocity * self._dt
            history = np.vstack((history, np.reshape(trajectory[t, :], (1, 2))))
        return trajectory

    @property
    def position(self) -> np.ndarray:
        return self._history[-1, :]

    @property
    def history(self) -> np.ndarray:
        return self._history

    @property
    def num_modes(self) -> int:
        return self._num_modes

    @abstractmethod
    def summary(self) -> Dict[str, Any]:
        summary = super(DTVObstacle, self).summary()
        summary.update({"history": self._history.tolist(), "num_modes": self._num_modes})
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(DTVObstacle, cls).from_summary(json_text)
        history = np.reshape(np.array(json_text["history"]), (-1, 2))
        summary.update({"history": history, "num_modes": json_text["num_modes"]})
        return summary
