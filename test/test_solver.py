import numpy as np

import mantrap.evaluation.scenarios as scenarios
from mantrap.solver import TrajOptSolver
from mantrap.utility.io import path_from_home_directory
from mantrap.visualization import plot_scene


###########################################################################
# Visualization ###########################################################
###########################################################################
def visualize_rrt_df_single_static():
    sim = scenarios.sf_ego_static_single_ado()
    solver = TrajOptSolver(sim, goal=np.array([5, 0.1]))
    ego_trajectory, ado_trajectories = solver.solve()

    output_dir = path_from_home_directory("test/graphs/solver_rrt_df_single_static")
    plot_scene(ado_trajectories, ado_colors=sim.ado_colors, ego_trajectory=ego_trajectory, output_dir=output_dir)


def visualize_rrt_df_single_moving():
    sim = scenarios.sf_ego_moving_single_ado()
    solver = TrajOptSolver(sim, goal=np.array([5, 0.1]))
    ego_trajectory, ado_trajectories = solver.solve()

    output_dir = path_from_home_directory("test/graphs/solver_rrt_df_single_moving")
    plot_scene(ado_trajectories, ado_colors=sim.ado_colors, ego_trajectory=ego_trajectory, output_dir=output_dir)


def visualize_rrt_df_multiple():
    sim = scenarios.sf_ego_moving_two_ados()
    solver = TrajOptSolver(sim, goal=np.array([5, 0.1]))
    ego_trajectory, ado_trajectories = solver.solve()

    output_dir = path_from_home_directory("test/graphs/solver_rrt_df_multiple")
    plot_scene(ado_trajectories, ado_colors=sim.ado_colors, ego_trajectory=ego_trajectory, output_dir=output_dir)


def visualize_rrt_df_really_multiple():
    sim = scenarios.sf_ego_moving_many_ados()
    solver = TrajOptSolver(sim, goal=np.array([5, 0.1]))
    ego_trajectory, ado_trajectories = solver.solve()

    output_dir = path_from_home_directory("test/graphs/solver_rrt_df_really_multiple")
    plot_scene(ado_trajectories, ado_colors=sim.ado_colors, ego_trajectory=ego_trajectory, output_dir=output_dir)


