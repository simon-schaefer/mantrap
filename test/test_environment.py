import matplotlib.pyplot as plt
import numpy as np

from murseco.environment.environment import Environment
from murseco.obstacle.cardinal import CardinalDiscreteTimeObstacle
from murseco.robot.cardinal import CardinalDiscreteTimeRobot
from murseco.utility.arrayops import rand_invsymmpos
import murseco.utility.io
from murseco.utility.visualization import plot_env_all_times


def test_environment_identifier():
    env = Environment((0, 10), (0, 10))
    for i in range(10):
        env.add_obstacle(CardinalDiscreteTimeObstacle(sigmas=rand_invsymmpos(4, 2, 2)))
    identifiers = [o.identifier for o in env.obstacles]
    assert len(np.unique(identifiers)) == 10


def test_environment_plot():
    env = Environment((-10, 10), (-10, 10))
    env.add_obstacle(CardinalDiscreteTimeObstacle(history=np.ones(2) * 3))
    env.add_obstacle(CardinalDiscreteTimeObstacle(history=np.ones(2) * (-4)))

    cache_path = murseco.utility.io.path_from_home_directory("test/cache/env_test.png")
    fig, ax = plt.subplots()
    murseco.utility.visualization.plot_env_at_time(fig, ax, env)
    plt.savefig(cache_path)


def test_environment_json():
    env_1 = Environment((-10, 10), (-10, 10))
    env_1.add_obstacle(CardinalDiscreteTimeObstacle(history=np.array([1.4, 4.2])))
    env_1.add_obstacle(CardinalDiscreteTimeObstacle(history=np.array([5.4, -2.94])))
    env_1.add_robot(CardinalDiscreteTimeRobot(np.array([1.31, 4.3]), 4, 1.0, np.ones((4, 1)) * 2))
    cache_path = murseco.utility.io.path_from_home_directory("test/cache/env_test.json")
    env_1.to_json(cache_path)
    env_2 = Environment.from_json(cache_path)
    assert env_1.summary() == env_2.summary()


def test_environment_visualization_all_times():
    env = Environment((-10, 10), (-10, 10))
    env.add_obstacle(CardinalDiscreteTimeObstacle(np.zeros(2), 1.0, np.array([np.eye(2) * 0.25] * 4), np.ones(4)))
    env.add_robot(CardinalDiscreteTimeRobot(np.array([1.31, 4.3]), 4, 1.0, np.ones((4, 1)) * 2))
    plot_env_all_times(env, murseco.utility.io.path_from_home_directory("test/cache/cardinal_visualization"))
