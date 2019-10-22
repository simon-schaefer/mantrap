import numpy as np
import pytest

import murseco.environment.scenarios
from murseco.planning import time_expanded_graph_search
from murseco.utility.io import path_from_home_directory
from murseco.utility.stats import Gaussian2D, GMM2D
from murseco.utility.visualization import plot_tppdf_trajectory


@pytest.mark.parametrize(
    "steps, grid_size, pos_start, pos_goal, risk_max", [(20, 100, np.array([-5, -2]), np.array([7.0, 3.0]), 0.005)]
)
def test_planninggraphsearch_static(
    steps: int, grid_size: int, pos_start: np.ndarray, pos_goal: np.ndarray, risk_max: float
):
    ppdf = GMM2D(np.array([[3.8, -1], [-3, 3]]), np.array([np.eye(2) * 1.0, np.eye(2) * 2.2]), weights=np.ones(2))
    x_grid, y_grid = np.meshgrid(np.linspace(-10, 10, grid_size), np.linspace(-10, 10, grid_size))
    tppdf = [ppdf.pdf_at(x_grid, y_grid)] * steps
    meshgrid = (x_grid, y_grid)
    trajectory, risk_sum = time_expanded_graph_search(pos_start, pos_goal, tppdf, meshgrid, risk_max)

    assert risk_sum <= risk_max
    assert trajectory.shape[0] <= steps
    assert np.isclose(np.linalg.norm(trajectory[0, :] - pos_start), 0)
    assert np.isclose(np.linalg.norm(trajectory[-1, :] - pos_goal), 0)

    dpath = path_from_home_directory("test/cache/graph_search_static")
    plot_tppdf_trajectory(tppdf, (x_grid, y_grid), dpath=dpath, rtrajectory=trajectory)


@pytest.mark.parametrize("risk_max", [0.0001])
def test_planninggraphsearch_static_none(risk_max: float):
    ppdf = Gaussian2D(np.array([0, 0]), np.eye(2))
    x_grid, y_grid = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    tppdf = [ppdf.pdf_at(x_grid, y_grid)] * 20
    pos_start, pos_goal = np.array([0.001, -0.05]), np.array([9, 9])
    trajectory, _ = time_expanded_graph_search(pos_start, pos_goal, tppdf, (x_grid, y_grid), risk_max)
    assert trajectory is None


@pytest.mark.parametrize(
    "risk_max, thorizon", [(0.01, 20), (0.1, 20), (1.0, 20)]
)
def test_planninggraphsearch_dynamic(risk_max: float, thorizon: int):
    env = murseco.environment.scenarios.vertical_fast(thorizon=thorizon)

    tppdf, (x_grid, y_grid) = env.tppdf()
    pos_start, pos_goal = np.array([-5, 0]), np.array([7, 0])
    trajectory, risk_sum = time_expanded_graph_search(pos_start, pos_goal, tppdf, (x_grid, y_grid), risk_max=risk_max)

    assert risk_sum <= risk_max
    assert trajectory.shape[0] <= thorizon
    assert np.isclose(np.linalg.norm(trajectory[0, :] - pos_start), 0)
    assert np.isclose(np.linalg.norm(trajectory[-1, :] - pos_goal), 0)

    dpath = path_from_home_directory(f"test/cache/graph_search_dynamic_{risk_max}")
    plot_tppdf_trajectory(tppdf, (x_grid, y_grid), dpath=dpath, rtrajectory=trajectory)
