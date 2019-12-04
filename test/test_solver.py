import numpy as np

from mantrap.agents import IntegratorDTAgent
from mantrap.simulation import DistanceFieldSimulation, SocialForcesSimulation
from mantrap.solver import TrajOptSolver
from mantrap.utility.io import path_from_home_directory
from mantrap.visualization import plot_scene


###########################################################################
# Visualization ###########################################################
###########################################################################
def visualize_rrt_df_single_static():
    sim = SocialForcesSimulation(
        ego_type=IntegratorDTAgent, ego_kwargs={"position": np.array([-5, 0.1]), "velocity": np.array([1, 0])},
    )
    sim.add_ado(position=np.zeros(2), velocity=np.zeros(2), goal_position=np.zeros(2))

    solver = TrajOptSolver(sim, goal=np.array([5, 0.1]))
    ego_trajectory, ado_trajectories = solver.solve()

    output_dir = path_from_home_directory("test/graphs/solver_rrt_df_single_static")
    plot_scene(ado_trajectories, ado_colors=sim.ado_colors, ego_trajectory=ego_trajectory, output_dir=output_dir)


def visualize_rrt_df_single_moving():
    sim = SocialForcesSimulation(
        ego_type=IntegratorDTAgent, ego_kwargs={"position": np.array([-5, 0.1]), "velocity": np.array([1, 0])},
    )
    sim.add_ado(position=np.zeros(2), velocity=np.array([0, 0.5]), goal_position=np.array([0, 5]))

    solver = TrajOptSolver(sim, goal=np.array([5, 0.1]))
    ego_trajectory, ado_trajectories = solver.solve()

    output_dir = path_from_home_directory("test/graphs/solver_rrt_df_single_moving")
    plot_scene(ado_trajectories, ado_colors=sim.ado_colors, ego_trajectory=ego_trajectory, output_dir=output_dir)


def visualize_rrt_df_multiple():
    sim = SocialForcesSimulation(
        ego_type=IntegratorDTAgent, ego_kwargs={"position": np.array([-5, 0.1]), "velocity": np.array([1, 0])},
    )
    sim.add_ado(position=np.array([0, 1]), velocity=np.array([-1, 0]), goal_position=np.ones(2)*(-10))
    sim.add_ado(position=np.array([1, -1]), velocity=np.array([-1, 0]), goal_position=np.ones(2)*(-10))

    solver = TrajOptSolver(sim, goal=np.array([5, 0.1]))
    ego_trajectory, ado_trajectories = solver.solve()

    output_dir = path_from_home_directory("test/graphs/solver_rrt_df_multiple")
    plot_scene(ado_trajectories, ado_colors=sim.ado_colors, ego_trajectory=ego_trajectory, output_dir=output_dir)


def visualize_rrt_df_really_multiple():
    sim = SocialForcesSimulation(
        ego_type=IntegratorDTAgent, ego_kwargs={"position": np.array([-5, 0.1]), "velocity": np.array([1, 0])},
    )
    sim.add_ado(position=np.array([0, 1]), velocity=np.array([-1, 0]), goal_position=np.ones(2)*(-10))
    sim.add_ado(position=np.array([1, -1]), velocity=np.array([-1, 0]), goal_position=np.ones(2)*(-10))
    sim.add_ado(position=np.array([5, -5]), velocity=np.array([2, 0.2]), goal_position=np.ones(2) * 10)
    sim.add_ado(position=np.array([3, 8]), velocity=np.array([-0.2, -0.5]), goal_position=np.array([2, -8]))
    sim.add_ado(position=np.array([7, -7]), velocity=np.array([-1, 1]), goal_position=np.array([-5, 5]))

    solver = TrajOptSolver(sim, goal=np.array([5, 0.1]))
    ego_trajectory, ado_trajectories = solver.solve()

    output_dir = path_from_home_directory("test/graphs/solver_rrt_df_really_multiple")
    plot_scene(ado_trajectories, ado_colors=sim.ado_colors, ego_trajectory=ego_trajectory, output_dir=output_dir)


