import time
from typing import Tuple, Union

import numpy as np
import pytest

from murseco.environment import Environment
from murseco.obstacle import StaticDTVObstacle
from murseco.robot import IntegratorDTRobot
from murseco.problem import D2TSProblem
import murseco.evaluation


class PseudoProblem:
    pass


def pseudo_planner(
    problem,
    trajectory: np.ndarray = np.zeros((20, 2)),
    controls: np.ndarray = np.zeros((20, 2)),
    risks: np.ndarray = np.ones(20) * 0.01,
    runtime: float = 0.0,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    time.sleep(runtime)
    return trajectory, controls, risks


def test_runtime():
    evaluation = murseco.evaluation.Evaluation(PseudoProblem(), pseudo_planner, runtime=0.005, reuse_results=False)
    runtime_mean = evaluation.results_mean[murseco.evaluation.COL_RUNTIME_MS]
    assert np.isclose(runtime_mean, 5, atol=2.0)


def test_d2ts_problem():
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

    assert all(
        [
            x in evaluation_mean_df.keys()
            for x in [
                murseco.evaluation.COL_TRAVEL_TIME_S,
                murseco.evaluation.COL_MIN_DISTANCE,
                murseco.evaluation.COL_RISK_EMPIRICAL,
                murseco.evaluation.COL_RUNTIME_MS,
            ]
        ]
    )
    assert evaluation_mean_df[murseco.evaluation.COL_RISK_EMPIRICAL] > 0
    assert 0 < evaluation_mean_df[murseco.evaluation.COL_TRAVEL_TIME_S] / problem.params.dt <= problem.params.thorizon


@pytest.mark.parametrize(
    "obstacle_trajectories, flag",
    [
        (np.zeros((2, 20, 2)), True),
        (np.ones((2, 20, 2)) * 0.05, True),
        (np.ones((5, 20, 2)) * 1.0, False),
        (np.reshape(np.vstack((np.linspace(-6, 4, num=20), np.zeros(20))).T, (1, 20, 2)), False),
    ],
)
def test_risk_standalone(obstacle_trajectories: np.ndarray, flag: bool):
    thorizon = 20
    robot_trajectory = np.vstack((np.linspace(-5, 5, num=thorizon), np.zeros(thorizon))).T
    is_colliding = murseco.evaluation.Evaluation.check_for_collision(
        robot_trajectory, obstacle_trajectories, max_collision_distance=0.1
    )
    assert is_colliding == flag


def test_robot_obstacle_distances():
    thorizon = 20
    robot_trajectory = np.vstack((np.linspace(-6, 6, num=thorizon), np.zeros(thorizon))).T

    obstacle_trajectories = np.zeros((2, thorizon, 2))
    obstacle_trajectories[0, :, :] = np.vstack((np.linspace(-6, 6, num=thorizon), np.ones(thorizon))).T
    obstacle_trajectories[1, :, :] = np.vstack((np.linspace(-5, 7, num=thorizon), np.zeros(thorizon))).T

    distances = murseco.evaluation.Evaluation.robot_obstacle_distances(robot_trajectory, obstacle_trajectories, 100)
    assert distances.shape == (2, 100)
    assert np.isclose(np.linalg.norm(distances[0, :].squeeze() - np.ones(100)), 0)
    assert np.isclose(np.linalg.norm(distances[1, :].squeeze() - np.ones(100)), 0)
    assert np.alltrue(distances >= 0)


@pytest.mark.parametrize(
    "trajectory, x_goal, num_steps",
    [
        (np.vstack((np.zeros(20), np.hstack((np.zeros(10), np.ones(10))))).T, np.array([0, 1]), 10),
        (np.zeros((20, 2)), np.ones(2), None),
    ],
)
def test_number_of_steps_standalone(trajectory: np.ndarray, x_goal: np.ndarray, num_steps: Union[int, None]):
    number_of_steps = murseco.evaluation.Evaluation.number_of_steps(trajectory, x_goal)
    assert number_of_steps == num_steps


def test_robot_obstacle_min_distance():
    thorizon = 20
    robot_trajectory = np.vstack((np.linspace(-5, 5, num=thorizon), np.zeros(thorizon))).T

    obstacle_trajectories = np.zeros((2, thorizon, 2))
    obstacle_trajectories[0, :, :] = np.vstack((np.linspace(-5, 5, num=thorizon), np.ones(thorizon))).T
    obstacle_trajectories[1, :, :] = np.vstack((np.linspace(-4, 6, num=thorizon), np.zeros(thorizon))).T
    obstacle_trajectories[0, 5, 1] = 0.01

    min_distance = murseco.evaluation.Evaluation.robot_obstacle_min_distance(robot_trajectory, obstacle_trajectories)
    assert np.isclose(min_distance, 0.01, atol=0.001)
