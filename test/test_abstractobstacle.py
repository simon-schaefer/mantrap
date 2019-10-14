import os
from typing import Any, Dict

import numpy as np
import pytest

from murseco.obstacle.abstract import DTVObstacle
import murseco.utility.io
from murseco.utility.stats import Point2D


class AbstractDTVObstacle(DTVObstacle):
    def __init__(self, history: np.ndarray, direction: np.ndarray = None, **kwargs):
        kwargs.update({"name": "test/test_abstractobstacle/TestDiscreteTimeObstacle", "num_modes": 1})
        super(AbstractDTVObstacle, self).__init__(history, **kwargs)
        self._direction = np.zeros(2) if direction is None else direction

    def vpdf(self, history: np.ndarray = None) -> Point2D:
        super(AbstractDTVObstacle, self).vpdf(history)
        return Point2D(self._direction)

    def vpdf_by_mode(self, mode: int, history: np.ndarray = None) -> Point2D:
        assert mode == 0, "object has only one mode"
        return self.vpdf()

    def summary(self) -> Dict[str, Any]:
        summary = super(AbstractDTVObstacle, self).summary()
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(AbstractDTVObstacle, cls).from_summary(json_text)
        return cls(**summary)


@pytest.mark.parametrize("direction", [np.random.rand(2), np.zeros(2), np.ones(2), np.array([2.9, -1.7])])
def test_abstractobstacle_trajectory(direction: np.ndarray):
    obstacle = AbstractDTVObstacle(np.zeros(2), direction=direction)
    thorizon = 4
    trajectories = obstacle.trajectory_samples(thorizon=thorizon, num_samples=4)
    assert trajectories.shape == (thorizon, 4, 2)
    assert all([np.array_equal(trajectories[0, :, :], trajectories[i, :, :]) for i in range(trajectories.shape[0])])
    trajectory = np.array([obstacle.position + i * direction for i in range(thorizon)])
    assert np.array_equal(trajectory, trajectories[0, :, :])


@pytest.mark.parametrize("direction", [np.random.rand(2), np.zeros(2), np.ones(2), np.array([2.9, -1.7])])
def test_abstractobstacle_ppdf(direction: np.ndarray):
    pinit, thorizon = np.zeros(2), 4
    obstacle = AbstractDTVObstacle(history=pinit, direction=direction)
    ppdf = obstacle.ppdf(thorizon)
    for t in range(thorizon):
        mu_expected = pinit + direction * t
        mu_ppdf = ppdf[t].gaussians[0].mu
        cov_ppdf = ppdf[t].gaussians[0].sigma
        assert np.isclose(np.linalg.norm(mu_expected - mu_ppdf), 0)
        assert np.isclose(np.abs(np.sum(cov_ppdf.flatten())), 0, atol=1e-3)


def test_abstractobstacle_json():
    obstacle_1 = AbstractDTVObstacle(np.array([1.31, 4.3]))
    cache_path = murseco.utility.io.path_from_home_directory("test/cache/abstractobstacle_test.json")
    obstacle_1.to_json(cache_path)
    obstacle_2 = AbstractDTVObstacle.from_json(cache_path)
    assert obstacle_1.summary() == obstacle_2.summary()
