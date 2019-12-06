import numpy as np

import mantrap.evaluation.scenarios as scenarios
from mantrap.evaluation.baselines import straight_line
from mantrap.solver import TrajOptSolver
from mantrap.evaluation import evaluate


###########################################################################
# Visualization ###########################################################
###########################################################################
def visualize_rrt_single_static():
    sim = scenarios.sf_ego_static_single_ado()
    solver = TrajOptSolver(sim, goal=np.array([5, 0.1]))
    ego_traj, ados_traj = solver.solve()
    evaluate("rrt_single_static", ego_traj, ados_traj, sim=sim, goal=np.array([5, 0.1]), baseline=straight_line)


def visualize_rrt_single_moving():
    sim = scenarios.sf_ego_moving_single_ado()
    solver = TrajOptSolver(sim, goal=np.array([5, 0.1]))
    ego_traj, ados_traj = solver.solve()
    evaluate("rrt_single_moving", ego_traj, ados_traj, sim=sim, goal=np.array([5, 0.1]), baseline=straight_line)


def visualize_rrt_some():
    sim = scenarios.sf_ego_moving_two_ados()
    solver = TrajOptSolver(sim, goal=np.array([5, 0.1]))
    ego_trajectory, ado_trajectories = solver.solve()
    evaluate("rrt_some", ego_trajectory, ado_trajectories, sim=sim, goal=np.array([5, 0.1]), baseline=straight_line)


def visualize_rrt_many():
    sim = scenarios.sf_ego_moving_many_ados()
    solver = TrajOptSolver(sim, goal=np.array([5, 0.1]))
    ego_trajectory, ado_trajectories = solver.solve()
    evaluate("rrt_many", ego_trajectory, ado_trajectories, sim=sim, goal=np.array([5, 0.1]), baseline=straight_line)
