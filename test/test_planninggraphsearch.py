from typing import List, Tuple

import numpy as np
import pytest

import murseco.environment.scenarios
from murseco.planning import time_expanded_graph_search
from murseco.utility.io import path_from_home_directory
from murseco.utility.stats import Gaussian2D, GMM2D
from murseco.utility.visualization import plot_tppdf_trajectory


@pytest.mark.parametrize("risk_max", [0.0001])
def test_planninggraphsearch_static_none(risk_max: float):
    ppdf = Gaussian2D(np.array([0, 0]), np.eye(2))
    x_grid, y_grid = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    tppdf = [ppdf.pdf_at(x_grid, y_grid)] * 20
    pos_start, pos_goal = np.array([0.001, -0.05]), np.array([9, 9])
    trajectory, _ = time_expanded_graph_search(pos_start, pos_goal, tppdf, (x_grid, y_grid), risk_max)
    assert trajectory is None


def plan_and_visualize(pos_start: np.ndarray, pos_goal: np.ndarray, tppdf: List[np.ndarray], meshgrid: Tuple[np.ndarray, np.ndarray], risk_max: float, dpath: str):
    x_grid, y_grid = meshgrid
    trajectory, risk_sum = time_expanded_graph_search(pos_start, pos_goal, tppdf, meshgrid, risk_max)

    assert risk_sum <= risk_max
    assert trajectory.shape[0] <= len(tppdf)
    assert np.isclose(np.linalg.norm(trajectory[0, :] - pos_start), 0)
    assert np.isclose(np.linalg.norm(trajectory[-1, :] - pos_goal), 0)

    plot_tppdf_trajectory(tppdf, (x_grid, y_grid), dpath=dpath, rtrajectory=trajectory)


def plan_env_and_visualize(pos_start: np.ndarray, pos_goal: np.ndarray, thorizon: int, risk_max: float, dpath: str):
    env = murseco.environment.scenarios.vertical_fast(thorizon=thorizon)
    tppdf, (x_grid, y_grid) = env.tppdf(num_points=200, mproc=False)

    plan_and_visualize(pos_start, pos_goal, tppdf, (x_grid, y_grid), risk_max, dpath=dpath)


def visualize_planninggraphsearch_static():
    steps, grid_size, pos_start, pos_goal, risk_max = 20, 100, np.array([-5, -2]), np.array([7.0, 3.0]), 0.005
    ppdf = GMM2D(np.array([[3.8, -1], [-3, 3]]), np.array([np.eye(2) * 1.0, np.eye(2) * 2.2]), weights=np.ones(2))
    x_grid, y_grid = np.meshgrid(np.linspace(-10, 10, grid_size), np.linspace(-10, 10, grid_size))
    dpath = path_from_home_directory(f"test/cache/graph_search_static_{risk_max}")

    tppdf = [ppdf.pdf_at(x_grid, y_grid)] * steps
    plan_and_visualize(pos_start, pos_goal, tppdf, (x_grid, y_grid), risk_max, dpath=dpath)


def visualize_planninggraphsearch_dynamic_001():
    pos_start, pos_goal = np.array([-5, 0]), np.array([7, 0])
    risk_max, thorizon = 0.01, 20
    dpath = path_from_home_directory(f"test/cache/graph_search_dynamic_001")

    plan_env_and_visualize(pos_start, pos_goal, thorizon, risk_max, dpath)


def visualize_planninggraphsearch_dynamic_010():
    pos_start, pos_goal = np.array([-5, 0]), np.array([7, 0])
    risk_max, thorizon = 0.1, 20
    dpath = path_from_home_directory(f"test/cache/graph_search_dynamic_010")

    plan_env_and_visualize(pos_start, pos_goal, thorizon, risk_max, dpath)


def visualize_planninggraphsearch_dynamic_100():
    pos_start, pos_goal = np.array([-5, 0]), np.array([7, 0])
    risk_max, thorizon = 1.0, 20
    dpath = path_from_home_directory(f"test/cache/graph_search_dynamic_100")

    plan_env_and_visualize(pos_start, pos_goal, thorizon, risk_max, dpath)

