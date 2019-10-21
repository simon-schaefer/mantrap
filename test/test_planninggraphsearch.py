import numpy as np

import murseco.planning.graph_search_x
from murseco.utility.io import path_from_home_directory
from murseco.utility.stats import Gaussian2D
from murseco.utility.visualization import plot_tppdf


x_size = y_size = 21  # 100
T = 20  # 20
pos_start, pos_goal = np.array([-5, -5]), np.array([5, 0])
x_grid, y_grid = np.meshgrid(np.linspace(-10, 10, x_size), np.linspace(-10, 10, y_size))
tppdf = [Gaussian2D(np.array([0, -4]), np.eye(2) * 4).pdf_at(x_grid, y_grid)] * T

trajectory, risk_sum = murseco.planning.graph_search_x.time_expanded_graph_search(pos_start, pos_goal, tppdf, (x_grid, y_grid))
print(trajectory)
print(risk_sum)

plot_tppdf(tppdf, (x_grid, y_grid), dpath=path_from_home_directory("test/cache/graph_search"), rtrajectory=trajectory)
