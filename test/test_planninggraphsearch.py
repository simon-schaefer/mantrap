import numpy as np
import pytest

import murseco.environment.scenarios
from murseco.environment import Environment
from murseco.planning import time_expanded_graph_search
from murseco.problem import D2TSProblem
from murseco.obstacle import SingleModeDTVObstacle, StaticDTVObstacle
from murseco.robot import IntegratorDTRobot
from murseco.utility.io import path_from_home_directory
from murseco.utility.visualization import plot_tppdf_trajectory


def plan_and_visualize(env: Environment, pos_goal: np.ndarray, risk_max: float, thorizon: int, dpath: str):
    problem = D2TSProblem(env, x_goal=pos_goal, risk_max=risk_max, thorizon=thorizon, mproc=False, grid_resolution=0.05)
    trajectory, acc_risks = time_expanded_graph_search(problem)

    assert acc_risks[-1] <= risk_max
    assert trajectory.shape[0] == problem.params.thorizon
    assert np.isclose(np.linalg.norm(trajectory[0, :] - problem.x_start_goal[0]), 0)
    assert np.isclose(np.linalg.norm(trajectory[-1, :] - problem.x_start_goal[1]), 0)

    tppdf, (x_grid, y_grid) = problem.grid
    titles = [f"acc. risk = {risk:.5f}" for risk in acc_risks]
    plot_tppdf_trajectory(tppdf, (x_grid, y_grid), dpath=dpath, rtrajectory=trajectory, titles=titles)


@pytest.mark.parametrize("risk_max, thorizon", [(0.005, 20)])
def test_planninggraphsearch_static_none(risk_max: float, thorizon: int):
    pos_start, pos_goal = np.array([-5, -2]), np.array([7.0, 3.0])

    env = Environment()
    env.add_obstacle(StaticDTVObstacle, mu=np.array([-3, 3]), covariance=np.eye(2) * 2.2)
    env.add_robot(IntegratorDTRobot, position=pos_start)

    problem = D2TSProblem(env, x_goal=pos_goal, risk_max=risk_max, thorizon=thorizon, mproc=False, grid_resolution=0.05)
    trajectory, acc_risks = time_expanded_graph_search(problem)

    assert acc_risks.size == problem.params.thorizon
    assert trajectory.shape[0] == problem.params.thorizon
    assert acc_risks[-1] <= risk_max
    assert np.isclose(np.linalg.norm(trajectory[0, :] - problem.x_start_goal[0]), 0)
    assert np.isclose(np.linalg.norm(trajectory[-1, :] - problem.x_start_goal[1]), 0)


def visualize_planninggraphsearch_static():
    steps, pos_start, pos_goal, risk_max = 20, np.array([-5, -2]), np.array([7.0, 3.0]), 0.005
    dpath = path_from_home_directory(f"test/graphs/graph_search_static_{risk_max}")

    env = Environment()
    env.add_obstacle(StaticDTVObstacle, mu=np.array([3.8, -1]), covariance=np.eye(2))
    env.add_obstacle(StaticDTVObstacle, mu=np.array([-3, 3]), covariance=np.eye(2) * 2.2)
    env.add_robot(IntegratorDTRobot, position=pos_start)
    plan_and_visualize(env, pos_goal, risk_max=risk_max, thorizon=steps, dpath=dpath)


def visualize_planninggraphsearch_dynamic_001():
    pos_start, pos_goal = np.array([-5, 0]), np.array([7, 0])
    dpath = path_from_home_directory(f"test/graphs/graph_search_dynamic_001")
    env = murseco.environment.scenarios.vertical_fast()
    env.add_robot(IntegratorDTRobot, position=pos_start)

    plan_and_visualize(env, pos_goal, risk_max=0.01, thorizon=20, dpath=dpath)


def visualize_planninggraphsearch_dynamic_010():
    pos_start, pos_goal = np.array([-5, 0]), np.array([7, 0])
    dpath = path_from_home_directory(f"test/graphs/graph_search_dynamic_010")
    env = murseco.environment.scenarios.vertical_fast()
    env.add_robot(IntegratorDTRobot, position=pos_start)

    plan_and_visualize(env, pos_goal, risk_max=0.1, thorizon=20, dpath=dpath)


def visualize_planninggraphsearch_dynamic_100():
    pos_start, pos_goal = np.array([-5, 0]), np.array([7, 0])
    dpath = path_from_home_directory(f"test/graphs/graph_search_dynamic_100")
    env = murseco.environment.scenarios.vertical_fast()
    env.add_robot(IntegratorDTRobot, position=pos_start)

    plan_and_visualize(env, pos_goal, risk_max=1.0, thorizon=20, dpath=dpath)
