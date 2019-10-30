import numpy as np
import pytest

from murseco.environment import Environment
from murseco.robot import IntegratorDTRobot
from murseco.problem import D2TSProblem
from murseco.utility.io import path_from_home_directory


@pytest.mark.xfail(raises=AssertionError)
def test_assert():
    env = Environment(xaxis=(-10, 10), yaxis=(-10, 10))
    env.add_robot(IntegratorDTRobot, position=np.array([25, 8]))
    D2TSProblem(env, x_goal=np.array([10, 10]))


def test_json():
    env = Environment()
    env.add_robot(IntegratorDTRobot)
    problem_1 = D2TSProblem(env, x_goal=np.array([9.1, -5.0]), w_x=2.15, w_u=102.9)
    cache_path = path_from_home_directory("test/cache/d2ts_test.json")
    problem_1.to_json(cache_path)
    problem_2 = D2TSProblem.from_json(cache_path)
    assert problem_1.summary() == problem_2.summary()
