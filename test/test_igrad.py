from pprint import pprint
import logging

from mantrap.agents import IntegratorDTAgent
from mantrap.simulation import SocialForcesSimulation
from mantrap.solver import IGradGreedySolver, IGradPredictiveSolver
from mantrap.evaluation import evaluate


# def visualize_interactive_grad_vector_field():
#     sim, goal = scenarios.scenario_sf_ego_moving_many_ados(ego_type=DoubleIntegratorDTAgent)
#     x_axis, y_axis = sim.axes
#     num_points = 50
#     x_grid, y_grid = np.meshgrid(
#         np.linspace(x_axis[0], x_axis[1], num_points), np.linspace(y_axis[0], y_axis[1], num_points)
#     )
#     vector_field = np.zeros((num_points, num_points, 4))
#     for ix in range(num_points):
#         for iy in range(num_points):
#             vector_field[ix, iy, 0:2] = np.array([x_grid[ix, iy], y_grid[ix, iy]])
#             vector_field[ix, iy, 2:4] = IGradGreedySolver.compute_interaction_grad(sim, vector_field[ix, iy, 0:2])
#             vector_field[ix, iy, 2:4] = -vector_field[ix, iy, 2:4]
#     vector_field = np.reshape(vector_field, (num_points ** 2, 4))
#
#     for k in range(num_points * num_points):
#         vector_length = np.linalg.norm(vector_field[k, 2:4])
#         vector_field[k, 2:4] = vector_field[k, 2:4] / vector_length * min(vector_length, 1)
#
#     import matplotlib.pyplot as plt
#
#     fig, ax = plt.subplots()
#     plt.quiver(x_grid, y_grid, vector_field[:, 2], vector_field[:, 3], units="xy", scale=2)
#     ax.grid(which="minor", alpha=0.1)
#     ax.grid(which="major", alpha=0.3)
#     for ado in sim.ados:
#         print(ado.position)
#     plt.show()
