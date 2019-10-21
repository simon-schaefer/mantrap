from typing import Tuple

import numpy as np

import murseco.planning.graph_search_x
from murseco.utility.io import path_from_home_directory
from murseco.utility.stats import GMM2D
from murseco.utility.visualization import plot_tppdf_trajectory


T = 20
x_size = y_size = 71
pos_start, pos_goal = np.array([-7, 0]), np.array([7.0, 3.0])
ppdf = GMM2D(np.array([[3.8, -1], [-3, 3]]), np.array([np.eye(2) * 1.6, np.eye(2) * 1.8]), weights=np.ones(2))

x_grid, y_grid = np.meshgrid(np.linspace(-10, 10, x_size), np.linspace(-10, 10, y_size))
tppdf = [ppdf.pdf_at(x_grid, y_grid)] * T
trajectory, risk_sum = murseco.planning.graph_search_x.time_expanded_graph_search(
    pos_start, pos_goal, tppdf, (x_grid, y_grid)
)

dpath = path_from_home_directory("test/cache/graph_search")
plot_tppdf_trajectory(tppdf, (x_grid, y_grid), dpath=dpath, rtrajectory=trajectory)
