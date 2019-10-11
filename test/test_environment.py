import numpy as np

from murseco.environment.environment import Environment
from murseco.obstacle.cardinal import CardinalDiscreteTimeObstacle
from murseco.robot.cardinal import CardinalDiscreteTimeRobot
from murseco.utility.arrayops import rand_invsymmpos
import murseco.utility.io
from murseco.utility.visualization import plot_env_initial_pdf, plot_env_samples


def test_environment_identifier():
    env = Environment((0, 10), (0, 10))
    for i in range(10):
        env.add_obstacle(CardinalDiscreteTimeObstacle(sigmas=rand_invsymmpos(4, 2, 2)))
    identifiers = [o.identifier for o in env.obstacles]
    assert len(np.unique(identifiers)) == 10


def test_environment_json():
    env_1 = Environment((-10, 10), (-10, 10))
    env_1.add_obstacle(CardinalDiscreteTimeObstacle(history=np.array([1.4, 4.2])))
    env_1.add_obstacle(CardinalDiscreteTimeObstacle(history=np.array([5.4, -2.94])))
    env_1.add_robot(CardinalDiscreteTimeRobot(np.array([1.31, 4.3]), 4, 1.0, np.ones((4, 1)) * 2))
    cache_path = murseco.utility.io.path_from_home_directory("test/cache/env_test.json")
    env_1.to_json(cache_path)
    env_2 = Environment.from_json(cache_path)
    assert env_1.summary() == env_2.summary()


def test_environment_visualization_samples():
    env = Environment((-10, 10), (-10, 10))
    env.add_obstacle(CardinalDiscreteTimeObstacle(sigmas=np.array([np.diag([1e-4, 1e-4])] * 4),
                                                  weights=np.array([2, 2, 1, 1])))
    env.add_robot(CardinalDiscreteTimeRobot(np.array([1.31, 4.3]), 4, 1.0, np.ones((4, 1)) * 2))
    plot_env_samples(env, murseco.utility.io.path_from_home_directory("test/cache/env_samples.png"))


def test_environment_visualization_initial():
    env = Environment((-10, 10), (-10, 10))
    env.add_obstacle(CardinalDiscreteTimeObstacle(velocity=2.0, sigmas=np.array([np.diag([1, 1])] * 4)))
    env.add_robot(CardinalDiscreteTimeRobot(position=np.array([4.53, 5.1])))
    plot_env_initial_pdf(env, murseco.utility.io.path_from_home_directory("test/cache/env_initial.png"))
