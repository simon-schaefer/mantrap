from abc import abstractmethod
from typing import Any, Dict

import numpy as np

from murseco.utility.io import JSONSerializer
from murseco.utility.stats import Distribution2D


class DiscreteTimeObstacle(JSONSerializer):
    def __init__(self, history: np.ndarray, **kwargs):
        super(DiscreteTimeObstacle, self).__init__(**kwargs)
        assert history.size % 2 == 0, "history array must consist of two-dimensional points"
        assert history.size > 0, "history array must not be empty"

        self._history = np.reshape(history, (-1, 2))  # obstacles last position (or initial position)

    @abstractmethod
    def pdf(self, history: np.ndarray = None) -> Distribution2D:
        """Given the obstacles history (trajectory) determine the probability density function of the next position."""
        return self._history.copy() if history is None else history

    def trajectory_samples(self, thorizon: int, num_samples: int) -> np.ndarray:
        """Sample possible (future) trajectories given the currently stored, thorizon steps in the future.
         Therefore iteratively call the pdf() method, sample the next position from that pdf, append the history and
         repeat until the time horizon (thorizon) is reached, for each trajectory sample.

         :argument thorizon: length of trajectory samples, i.e. number of predicted time-steps.
         :argument num_samples: number of trajectory samples.
         :returns array of sampled future trajectories (num_samples, thorizon, 2).
         """
        trajectories = []
        for i in range(num_samples):
            history = self._history.copy()
            trajectory = np.zeros((thorizon, 2))
            for t in range(thorizon):
                pdf = self.pdf(history)
                trajectory[t, :] = pdf.sample(1)
                history = np.vstack((history, np.reshape(trajectory[t, :], (1, 2))))
            trajectories.append(trajectory)
        return np.array(trajectories)

    @property
    def position(self) -> np.ndarray:
        return self._history[-1, :]

    @property
    def history(self) -> np.ndarray:
        return self._history

    @abstractmethod
    def summary(self) -> Dict[str, Any]:
        summary = super(DiscreteTimeObstacle, self).summary()
        summary.update({"history": self._history.tolist()})
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(DiscreteTimeObstacle, cls).from_summary(json_text)
        history = np.reshape(np.array(json_text["history"]), (-1, 2))
        summary.update({"history": history})
        return summary
