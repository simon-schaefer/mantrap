import numpy as np

from murseco.environment import Environment
from murseco.obstacle import SingleModeDTVObstacle
from murseco.robot import CardinalDTRobot
from murseco.utility.array import rand_invsymmpos
import murseco.utility.io
from murseco.utility.visualization import plot_trajectory_samples


def test_environment_identifier():
    env = Environment((0, 10), (0, 10))
    for i in range(10):
        env.add_obstacle(SingleModeDTVObstacle, covariance=rand_invsymmpos(2, 2))
    identifiers = [o.identifier for o in env.obstacles]
    assert len(np.unique(identifiers)) == 10


def test_environment_json():
    env_1 = Environment((-10, 10), (-10, 10), thorizon=4)
    env_1.add_obstacle(SingleModeDTVObstacle, history=np.array([1.4, 4.2]))
    env_1.add_obstacle(SingleModeDTVObstacle, history=np.array([5.4, -2.94]))
    env_1.add_robot(CardinalDTRobot, position=np.array([1.31, 4.3]), velocity=1.0, policy=np.ones((4, 1)) * 2)
    cache_path = murseco.utility.io.path_from_home_directory("test/cache/env_test.json")
    env_1.to_json(cache_path)
    env_2 = Environment.from_json(cache_path)
    assert env_1.summary() == env_2.summary()
    assert [x in env_1.summary().keys() for x in ["obstacles", "xaxis", "yaxis", "robot"]]
