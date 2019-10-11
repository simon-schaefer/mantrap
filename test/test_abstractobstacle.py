from typing import Any, Dict

import numpy as np
import pytest

from murseco.obstacle.abstract import DiscreteTimeObstacle
import murseco.utility.io
from murseco.utility.stats import Distribution2D, Point2D


class AbstractDiscreteTimeObstacle(DiscreteTimeObstacle):
    def __init__(self, history: np.ndarray, direction: np.ndarray = None, **kwargs):
        kwargs.update({"name": "test/test_abstractobstacle/TestDiscreteTimeObstacle"})
        super(AbstractDiscreteTimeObstacle, self).__init__(history, **kwargs)
        self._direction = np.zeros(2) if direction is None else direction

    def pdf(self, history: np.ndarray = None) -> Distribution2D:
        history = super(AbstractDiscreteTimeObstacle, self).pdf(history)
        return Point2D(history[-1, :] + self._direction)

    def summary(self) -> Dict[str, Any]:
        summary = super(AbstractDiscreteTimeObstacle, self).summary()
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(AbstractDiscreteTimeObstacle, cls).from_summary(json_text)
        return cls(**summary)


@pytest.mark.parametrize(
    "direction",
    [np.random.rand(2), np.zeros(2), np.ones(2), np.array([2.9, -1.7])]
)
def test_abstractobstacle_trajectory(direction: np.ndarray):
    obstacle = AbstractDiscreteTimeObstacle(np.zeros(2), direction=direction)
    thorizon = 4
    trajectories = obstacle.trajectory_samples(thorizon=thorizon, num_samples=4)
    assert trajectories.shape == (thorizon, 4, 2)
    assert all([np.array_equal(trajectories[0, :, :], trajectories[i, :, :]) for i in range(trajectories.shape[0])])
    trajectory = np.array([obstacle.position + (i + 1) * direction for i in range(thorizon)])
    assert np.array_equal(trajectory, trajectories[0, :, :])


def test_abstractobstacle_json():
    obstacle_1 = AbstractDiscreteTimeObstacle(np.array([1.31, 4.3]))
    cache_path = murseco.utility.io.path_from_home_directory("test/cache/abstractobstacle_test.json")
    obstacle_1.to_json(cache_path)
    obstacle_2 = AbstractDiscreteTimeObstacle.from_json(cache_path)
    assert obstacle_1.summary() == obstacle_2.summary()
