from itertools import product
import logging
from pprint import pprint
import time
from typing import Any, Dict, Tuple

import torch

from mantrap.agents import DoubleIntegratorDTAgent
from mantrap.evaluation import evaluate_metrics
from mantrap.environment.environment import GraphBasedEnvironment
from mantrap.environment import PotentialFieldEnvironment, Trajectron
from mantrap.solver.solver import Solver
from mantrap.solver import SGradSolver, MonteCarloTreeSearch
from mantrap.utility.io import load_functions_from_module


###########################################################################
# Solvers #################################################################
###########################################################################
# def solver_sgrad_t5() -> Tuple[Solver.__class__, Dict[str, Any]]:
#     return SGradSolver, {"t_planning": 5}
#
#
# def solver_sgrad_t2() -> Tuple[Solver.__class__, Dict[str, Any]]:
#     return SGradSolver, {"t_planning": 2}
# #
#
# # def solver_igrad_t5() -> Tuple[Solver.__class__, Dict[str, Any]]:
# #     return IGradSolver, {"t_planning": 5}
#
#
def solver_mcts_t5() -> Tuple[Solver.__class__, Dict[str, Any]]:
    return MonteCarloTreeSearch, {"t_planning": 5, "config_name": "t5"}


# def solver_mcts_t1() -> Tuple[Solver.__class__, Dict[str, Any]]:
#     return MonteCarloTreeSearch, {"t_planning": 1}


# def solver_orca() -> Tuple[Solver.__class__, Dict[str, Any]]:
#     return ORCASolver, {"t_planning": 1}


###########################################################################
# Scenarios ###############################################################
###########################################################################
# def scenario_haruki() -> Tuple[GraphBasedSimulation.__class__, torch.Tensor]:
#     ego_position = torch.tensor([-7, 0])
#     ego_velocity = torch.zeros(2)
#     ego_goal = torch.tensor([7, -1])
#     ado_positions = torch.stack((torch.tensor([-7, -1]), torch.tensor([7, 3]), torch.tensor([7, -2])))
#     ado_goals = torch.stack((torch.tensor([0, 0]), torch.tensor([-7, 0]), torch.tensor([-7, 4])))
#     ado_velocities = torch.stack((torch.tensor([1, 0]), torch.tensor([-1, 0]), torch.tensor([-1, 1])))
#
#     ego_kwargs = {"position": ego_position, "velocity": ego_velocity}
#     env = PotentialFieldEnvironment(DoubleIntegratorDTAgent, ego_kwargs, scene_name="haruki")
#     for position, ado_goal, velocity in zip(ado_positions, ado_goals, ado_velocities):
#         env.add_ado(position=position, goal=ado_goal, velocity=velocity, num_modes=1)
#     return env, ego_goal


def scenario_haruki_trajectron() -> Tuple[GraphBasedEnvironment.__class__, torch.Tensor]:
    ego_position = torch.tensor([-7, 0])
    ego_velocity = torch.zeros(2)
    ego_goal = torch.tensor([7, -1])
    ado_positions = torch.stack((torch.tensor([-7, -1]), torch.tensor([7, 3]), torch.tensor([7, -2])))
    ado_goals = torch.stack((torch.tensor([0, 0]), torch.tensor([-7, 0]), torch.tensor([-7, 4])))
    ado_velocities = torch.stack((torch.tensor([1, 0]), torch.tensor([-1, 0]), torch.tensor([-1, 1])))

    ego_kwargs = {"position": ego_position, "velocity": ego_velocity}
    env = Trajectron(DoubleIntegratorDTAgent, ego_kwargs, scene_name="haruki_traj")
    for position, velocity in zip(ado_positions, ado_velocities):
        env.add_ado(position=position, velocity=velocity, num_modes=1)
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
        name = f"{solver_name}:{scenario_name}"

        # Initialise environment environment and solver as described in functions.
        logging.warning(f"Evaluation: solver {solver_name} in scenario {scenario_name}")
        sim, goal = scenario_func()
        solver_class, solver_kwargs = solver_func()
        solver = solver_class(sim, goal=goal, verbose=2, multiprocessing=True, **solver_kwargs)

        # Solve posed problem, until goal has been reached.
        config_kwargs = {"time_steps": 10, "max_cpu_time": 0.5}
        x_opt, ado_traj = solver.solve(**config_kwargs)

        # Log and visualise results for later comparison.
        results[name] = evaluate_metrics(ego_trajectory=x_opt, ado_trajectories=ado_traj, env=sim, goal=goal)
        logging.warning(f"Evaluation ==> {results[name]}")

    logging.warning("Evaluation results:")
    time.sleep(0.2)  # wait until loop has finished
    pprint(results)
