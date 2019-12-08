import logging

from mantrap.agents import DoubleIntegratorDTAgent
from mantrap.evaluation.baselines import straight_line
from mantrap.solver import IGradGreedySolver, IGradPredictiveSolver
from mantrap.evaluation import evaluate, scenarios


###########################################################################
# Visualization ###########################################################
###########################################################################
def visualize_rrt_greedy():
    for tag, scenario in scenarios().items():
        logging.info(f"Test scenario: {tag}")
        sim, goal = scenario(ego_type=DoubleIntegratorDTAgent)
        solver = IGradGreedySolver(sim, goal=goal)
        ego_traj, ados_traj = solver.solve()
        evaluate(f"igrad_greedy_{tag}", ego_traj, ados_traj, sim=sim, goal=goal, baseline=straight_line)


def visualize_rrt_predictive():
    for tag, scenario in scenarios().items():
        logging.info(f"Test scenario: {tag}")
        sim, goal = scenario(ego_type=DoubleIntegratorDTAgent)
        solver = IGradPredictiveSolver(sim, goal=goal)
        ego_traj, ados_traj = solver.solve()
        evaluate(f"igrad_pred_{tag}", ego_traj, ados_traj, sim=sim, goal=goal, baseline=straight_line)
