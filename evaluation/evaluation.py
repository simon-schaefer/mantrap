from itertools import product
import logging
from pprint import pprint
import time
from typing import Any, Dict, Tuple

import torch

from mantrap.agents import DoubleIntegratorDTAgent
from mantrap.evaluation import evaluate_metrics
from mantrap.evaluation.visualization import visualize_scenes
from mantrap.simulation.simulation import GraphBasedSimulation
from mantrap.simulation import PotentialFieldSimulation
from mantrap.solver.solver import Solver
from mantrap.solver import IGradSolver, SGradSolver, MonteCarloTreeSearch
from mantrap.utility.io import build_os_path, load_functions_from_module


###########################################################################
# Solvers #################################################################
###########################################################################
def solver_sgrad_t5() -> Tuple[Solver.__class__, Dict[str, Any]]:
    return SGradSolver, {"t_planning": 5}


def solver_sgrad_t2() -> Tuple[Solver.__class__, Dict[str, Any]]:
    return SGradSolver, {"t_planning": 2}


def solver_igrad_t5() -> Tuple[Solver.__class__, Dict[str, Any]]:
    return IGradSolver, {"t_planning": 5}


def solver_mcts_t5() -> Tuple[Solver.__class__, Dict[str, Any]]:
    return MonteCarloTreeSearch, {"t_planning": 5}


def solver_mcts_t1() -> Tuple[Solver.__class__, Dict[str, Any]]:
    return MonteCarloTreeSearch, {"t_planning": 1}


# def solver_orca() -> Tuple[Solver.__class__, Dict[str, Any]]:
#     return ORCASolver, {"t_planning": 1}


###########################################################################
# Scenarios ###############################################################
###########################################################################
# TODO: Evaluation simulation integration
def scenario_haruki() -> Tuple[GraphBasedSimulation.__class__, torch.Tensor]:
    ego_position = torch.tensor([-7, 0])
    ego_velocity = torch.zeros(2)
    ego_goal = torch.tensor([7, -1])
    ado_positions = torch.stack((torch.tensor([-7, -1]), torch.tensor([7, 3]), torch.tensor([7, -2])))
    ado_goals = torch.stack((torch.tensor([0, 0]), torch.tensor([-7, 0]), torch.tensor([-7, 4])))
    ado_velocities = torch.stack((torch.tensor([1, 0]), torch.tensor([-1, 0]), torch.tensor([-1, 1])))

    env = PotentialFieldSimulation(DoubleIntegratorDTAgent, {"position": ego_position, "velocity": ego_velocity})
    for position, ado_goal, velocity in zip(ado_positions, ado_goals, ado_velocities):
        env.add_ado(position=position, goal=ado_goal, velocity=velocity, num_modes=1)
    return env, ego_goal


# def scenario_haruki_si() -> Tuple[GraphBasedSimulation.__class__, torch.Tensor]:
#     ego_position = torch.tensor([-7, 0])
#     ego_velocity = torch.zeros(2)
#     ego_goal = torch.tensor([7, -1])
#     ado_positions = torch.stack((torch.tensor([-7, -1]), torch.tensor([7, 3]), torch.tensor([7, -2])))
#     ado_goals = torch.stack((torch.tensor([0, 0]), torch.tensor([-7, 0]), torch.tensor([-7, 4])))
#     ado_velocities = torch.stack((torch.tensor([1, 0]), torch.tensor([-1, 0]), torch.tensor([-1, 1])))
#
#     env = PotentialFieldSimulation(IntegratorDTAgent, {"position": ego_position, "velocity": ego_velocity})
#     for position, ado_goal, velocity in zip(ado_positions, ado_goals, ado_velocities):
#         env.add_ado(position=position, goal=ado_goal, velocity=velocity, num_modes=1)
#     return env, ego_goal


###########################################################################
# Script ##################################################################
###########################################################################
if __name__ == '__main__':
    scenarios = load_functions_from_module(module="evaluation", prefix="scenario_")
    solvers = load_functions_from_module(module="evaluation", prefix="solver_")

    results = {}
    for (scenario_name, scenario_func), (solver_name, solver_func) in product(scenarios.items(), solvers.items()):
        logging.warning(f"Evaluation: solver {solver_name} in scenario {scenario_name}")
        sim, goal = scenario_func()
        solver_class, solver_kwargs = solver_func()
        solver = solver_class(sim, goal=goal, verbose=False, **solver_kwargs)

        x_opt, ado_traj, x_planned, ado_planned = solver.solve(time_steps=20, max_cpu_time=0.5, multiprocessing=True)

        exp_tag = f"{solver_name}:{scenario_name}"
        results[exp_tag] = evaluate_metrics(ego_trajectory=x_opt, ado_trajectories=ado_traj, env=sim, goal=goal)
        visualize_scenes(x_planned, ado_planned, env=sim, file_path=build_os_path(f"test/graphs/{exp_tag}"))
        logging.warning(f"Evaluation ==> {results[exp_tag]}")

    logging.warning("Evaluation results:")
    time.sleep(0.2)  # wait until loop has finished
    pprint(results)
