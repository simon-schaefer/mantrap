from typing import List, Tuple

import numpy as np
import pytest

import murseco.planning.mcstatic
from murseco.environment import Environment
import murseco.environment.scenarios
from murseco.obstacle import StaticDTVObstacle
from murseco.robot import IntegratorDTRobot
from murseco.problem import D2TSProblem
from murseco.utility.io import path_from_home_directory
from murseco.utility.visualization import plot_tppdf_trajectory


@pytest.mark.parametrize(
    "array, safe_intervals_expected",
    [
        (np.zeros(10, dtype=bool), [(0, 10)]),
        (np.hstack((np.zeros(2, dtype=bool), np.ones(8, dtype=bool))), [(0, 2)]),
        (np.hstack((np.ones(2, dtype=bool), np.zeros(3, dtype=bool), np.ones(5, dtype=bool))), [(2, 5)]),
    ],
)
def test_compute_safe_intervals(array: np.ndarray, safe_intervals_expected: List[Tuple]):
    static_map = np.reshape(array, (10, 1, 1))
    safe_intervals = murseco.planning.mcstatic.compute_safe_intervals(static_map)
    assert safe_intervals == [[safe_intervals_expected]]


def test_trajectory_flatten():
    trajectory_3d = np.array([[0, 0, 0], [4, 1, 2], [6, 6, 3]])
    trajectory_2d = murseco.planning.mcstatic.trajectory_flatten(trajectory_3d)
    trajectory_2d_exp = np.array([[0, 0], [0, 0], [4, 1], [6, 6]])
    assert trajectory_2d.shape == trajectory_2d_exp.shape
    assert np.array_equal(trajectory_2d, trajectory_2d_exp)


def tppdf_to_static_map(tppdf: List[np.ndarray], risk_max: float) -> np.ndarray:
    static_map = np.array(tppdf)
    static_map[static_map < risk_max] = False
    static_map[static_map >= risk_max] = True
    return static_map


def plan_and_visualize(env: Environment, risk_max: float, dpath: str):
    env.add_robot(IntegratorDTRobot, position=np.array([-9, 5]))
    problem = D2TSProblem(env, x_goal=np.array([9, -2]), thorizon=300, dt=0.1, grid_resolution=0.2, risk_max=risk_max)

    trajectory, _, _, static_map = murseco.planning.monte_carlo_static(problem, return_static_maps=True)

    tppdf = [static_map[t, :, :].astype(float) for t in range(static_map.shape[0])]
    _, (x_grid, y_grid) = problem.grid
    plot_tppdf_trajectory(tppdf, (x_grid, y_grid), dpath=dpath, rtrajectory=trajectory)


def visualize_safe_interval_a_star_static():
    env = murseco.environment.scenarios.single_static(dt=0.1)
    plan_and_visualize(env, risk_max=0.8, dpath=path_from_home_directory(f"test/cache/safe_interval_a_star_static"))


def visualize_safe_interval_a_star_dynamic():
    env = murseco.environment.scenarios.double_two_mode(dt=0.1)
    plan_and_visualize(env, risk_max=0.001, dpath=path_from_home_directory(f"test/cache/safe_interval_a_star_dynamic"))
