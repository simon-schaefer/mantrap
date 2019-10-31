import numpy as np
import pytest

from murseco.environment import Environment
from murseco.obstacle import StaticDTVObstacle
from murseco.robot import IntegratorDTRobot
from murseco.problem import DTCSProblem
from murseco.utility.io import path_from_home_directory
from murseco.utility.stats import GMM2D


@pytest.mark.xfail(raises=AssertionError)
def test_assert():
    env = Environment(xaxis=(-10, 10), yaxis=(-10, 10))
    env.add_robot(IntegratorDTRobot, position=np.array([25, 8]))
    DTCSProblem(env, x_goal=np.array([10, 10]))


def test_ppdf_single():
    env = Environment(xaxis=(-10, 10), yaxis=(-10, 10))
    env.add_obstacle(StaticDTVObstacle, mu=np.array([-3, 3]), covariance=np.eye(2) * 2.2)
    env.add_robot(IntegratorDTRobot, position=np.array([5, 5]))
    problem = DTCSProblem(env, x_goal=np.array([10, 10]))

    assert len(problem.tppdf) == problem.params.thorizon
    assert np.alltrue([type(ppdf) == GMM2D for ppdf in problem.tppdf])


def test_ppdf_multiple():
    env = Environment(xaxis=(-20, 20), yaxis=(-20, 20))
    env.add_obstacle(StaticDTVObstacle, mu=np.array([-2, 3]), covariance=np.eye(2) * 2.2)
    env.add_obstacle(StaticDTVObstacle, mu=np.array([-1, 9]), covariance=np.eye(2) * 3.1)
    env.add_robot(IntegratorDTRobot, position=np.array([0, 0]))
    problem = DTCSProblem(env, x_goal=np.array([10, 10]))

    assert len(problem.tppdf) == problem.params.thorizon
    assert np.alltrue([type(ppdf) == GMM2D for ppdf in problem.tppdf])

    for ppdf in problem.tppdf:
        assert np.array_equal(ppdf.mus, np.array([[-2, 3], [-1, 9]]))
        assert np.array_equal(ppdf.weights, np.array([0.5, 0.5]))  # default weight is 1


def test_json():
    env = Environment()
    env.add_robot(IntegratorDTRobot)
    problem_1 = DTCSProblem(env, x_goal=np.array([9.1, -5.0]), w_x=2.15, w_u=102.9)
    cache_path = path_from_home_directory("test/cache/dtcs_test.json")
    problem_1.to_json(cache_path)
    problem_2 = DTCSProblem.from_json(cache_path)
    assert problem_1.summary() == problem_2.summary()
