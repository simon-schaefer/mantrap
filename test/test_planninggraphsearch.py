import numpy as np

from murseco.planning import time_expanded_graph_search
from murseco.utility.io import path_from_home_directory
from murseco.utility.stats import Gaussian2D
from murseco.utility.visualization import plot_tppdf


def test_planning_time_expanded_graph_search():
    x_size = y_size = 21  # 100
    T = 20  # 20
    pos_start, pos_goal = np.array([-5, 0]), np.array([8, 1])
    x_grid, y_grid = np.meshgrid(np.linspace(-10, 10, x_size), np.linspace(-10, 10, y_size))
    tppdf = [Gaussian2D(np.array([0, -4]), np.eye(2) * 4).pdf_at(x_grid, y_grid)] * T

    def cost(x: np.ndarray, u: np.ndarray) -> float:
        return np.linalg.norm(x - pos_goal) ** 2 + np.linalg.norm(u) ** 2

    def dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return x + u

    trajectory, risks, graph = time_expanded_graph_search(
        pos_start=pos_start,
        pos_goal=pos_goal,
        tppdf=tppdf,
        costs=cost,
        dynamics=dynamics,
        meshgrid=(x_grid, y_grid),
    )
    print(risks)
    print(np.sum(risks))
    plot_tppdf(tppdf, (x_grid, y_grid), path_from_home_directory("test/cache/graph_search_no_pdf"), trajectory)


if __name__ == "__main__":
    test_planning_time_expanded_graph_search()
