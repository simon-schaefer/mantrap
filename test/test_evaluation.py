import time
from typing import Tuple

import numpy as np

from murseco.environment import Environment
from murseco.obstacle import StaticDTVObstacle
from murseco.robot import IntegratorDTRobot
from murseco.problem import D2TSProblem
import murseco.evaluation


class PseudoProblem:
    pass


def pseudo_planner(
    problem, trajectory: np.ndarray = np.zeros((20, 2)), risks: np.ndarray = np.ones(20) * 0.01, runtime: float = 0.0, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    time.sleep(runtime)
    return trajectory, risks


def test_evaluation_runtime():
    evaluation = murseco.evaluation.Evaluation(
        PseudoProblem(), pseudo_planner, runtime=0.005, reuse_results=False
    )
    runtime_mean = evaluation.results_mean[murseco.evaluation.COL_RUNTIME_MS]
    assert np.isclose(runtime_mean, 5, atol=2.0)


def test_evaluation_risk():
    np.random.seed(0)
    env = Environment()
    env.add_obstacle(StaticDTVObstacle, mu=np.array([0, 0]), covariance=np.eye(2) * 0.01)
    env.add_obstacle(StaticDTVObstacle, mu=np.array([-10, 0]), covariance=np.eye(2) * 0.01)
    env.add_robot(IntegratorDTRobot, position=np.array([-5, 0]))
    problem = D2TSProblem(env, x_goal=np.array([5, 0]), mproc=False)

    thorizon = problem.params.thorizon
    planner_trajectory = np.vstack((np.linspace(-5, 5, num=thorizon), np.zeros(thorizon))).T

    evaluation = murseco.evaluation.Evaluation(problem, pseudo_planner, trajectory=planner_trajectory)
    evaluation_mean_df = evaluation.results_mean

    assert evaluation_mean_df[murseco.evaluation.COL_RISK_EMPIRICAL] > 0
