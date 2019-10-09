import matplotlib.pyplot as plt
import numpy as np
import pytest

from murseco.environment.environment import Environment
from murseco.obstacle.cardinal import CardinalDiscreteTimeObstacle
from murseco.obstacle.tgmm import TGMMDiscreteTimeObstacle
from murseco.utility.arrayops import rand_invsymmpos
import murseco.utility.io
import murseco.utility.visualization


def test_environment_identifier():
    env = Environment((0, 10), (0, 10))
    identifiers = []
    for i in range(10):
        tmus, tsigmas, tweights = np.ones((1, 2, 2)), rand_invsymmpos(1, 2, 2, 2), np.ones((1, 2))
        identifier = env.add_discrete_time_obstacle(TGMMDiscreteTimeObstacle(tmus, tsigmas, tweights))
        identifiers.append(identifier)
    assert len(np.unique(identifiers)) == 10


def test_environment_plot():
    env = Environment((-10, 10), (-10, 10))
    tsigmas, tweights = np.tile(np.eye(2), (1, 2, 1, 1)), np.ones((1, 2))
    env.add_discrete_time_obstacle(TGMMDiscreteTimeObstacle(np.ones((1, 2, 2)) * 3, tsigmas, tweights))
    env.add_discrete_time_obstacle(TGMMDiscreteTimeObstacle(np.ones((1, 2, 2)) * (-4), tsigmas, tweights))

    cache_path = murseco.utility.io.path_from_home_directory("test/cache/env_test.png")
    fig, ax = plt.subplots()
    murseco.utility.visualization.plot_env_at_time(fig, ax, env)
    plt.savefig(cache_path)


def test_environment_json():
    env_1 = Environment((-10, 10), (-10, 10))
    tsigmas, tweights = np.tile(np.eye(2), (1, 2, 1, 1)), np.ones((1, 2))
    env_1.add_discrete_time_obstacle(TGMMDiscreteTimeObstacle(np.ones((1, 2, 2)) * 5.6, tsigmas, tweights))
    env_1.add_discrete_time_obstacle(TGMMDiscreteTimeObstacle(np.ones((1, 2, 2)) * (-2.3), tsigmas, tweights))
    cache_path = murseco.utility.io.path_from_home_directory("test/cache/env_test.json")
    env_1.to_json(cache_path)
    env_2 = Environment.from_json(cache_path)
    assert env_1.summary() == env_2.summary()


@pytest.mark.xfail(raises=AssertionError)
def test_environment_same_tmax():
    env = Environment((-10, 10), (-10, 10))
    pinit, pstep, sigmas, weights = np.zeros(2), 5, rand_invsymmpos(4, 2, 2) * 0.01, np.ones(4)
    env.add_discrete_time_obstacle(CardinalDiscreteTimeObstacle(pinit, pstep, 5, sigmas, weights))
    env.add_discrete_time_obstacle(CardinalDiscreteTimeObstacle(pinit, pstep, 3, sigmas, weights))
