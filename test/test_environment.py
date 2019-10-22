import numpy as np

from murseco.environment import Environment
import murseco.environment.scenarios
from murseco.obstacle import SingleModeDTVObstacle
from murseco.robot import IntegratorDTRobot
from murseco.utility.array import rand_invsymmpos
from murseco.utility.io import path_from_home_directory
from murseco.utility.visualization import plot_trajectory_samples, plot_tppdf_trajectory


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
    env_1.add_robot(IntegratorDTRobot, position=np.array([1.31, 4.3]))
    cache_path = path_from_home_directory("test/cache/env_test.json")
    env_1.to_json(cache_path)
    env_2 = Environment.from_json(cache_path)
    assert env_1.summary() == env_2.summary()
    assert [x in env_1.summary().keys() for x in ["obstacles", "xaxis", "yaxis", "robot"]]


def visualize_environment_scenario_scenario_samples():
    env = murseco.environment.scenarios.double_two_mode(thorizon=10)

    otrajectory_samples = env.generate_trajectory_samples()
    ohistories = [o.history for o in env.obstacles]
    ocolors = [o.color for o in env.obstacles]

    fpath = path_from_home_directory("test/cache/scenario_double_two.png")
    plot_trajectory_samples(otrajectory_samples, ohistories, ocolors, xaxis=env.xaxis, fpath=fpath)


def visualize_environment_tppdf():
    env = murseco.environment.scenarios.double_two_mode(thorizon=20)
    robot_pinit = np.array([-5, 0])
    env.add_robot(IntegratorDTRobot, position=robot_pinit)

    tppdf, meshgrid = env.tppdf(num_points=200, mproc=False)

    dpath = path_from_home_directory("test/cache/scenario_double_two_mode")
    plot_tppdf_trajectory(tppdf, meshgrid, dpath=dpath)
