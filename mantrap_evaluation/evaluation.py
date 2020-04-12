from itertools import product
import logging
from pprint import pprint
import time
from typing import Any, Dict, Tuple

from mantrap.evaluation import evaluate_metrics
from mantrap.solver.solver import Solver
from mantrap.solver import IGradSolver, ORCASolver, SGradSolver, MonteCarloTreeSearch
from mantrap.utility.io import load_functions_from_module


###########################################################################
# Solvers #################################################################
###########################################################################
def solver_sgrad_t5() -> Tuple[Solver.__class__, Dict[str, Any]]:
    return SGradSolver, {"t_planning": 5, "config_name": "t5"}


# def solver_sgrad_t2() -> Tuple[Solver.__class__, Dict[str, Any]]:
#     return SGradSolver, {"t_planning": 2, "config_name": "t2"}


# def solver_igrad_t5() -> Tuple[Solver.__class__, Dict[str, Any]]:
#     return IGradSolver, {"t_planning": 5, "config_name": "t5"}
#
#
# def solver_mcts_t5() -> Tuple[Solver.__class__, Dict[str, Any]]:
#     return MonteCarloTreeSearch, {"t_planning": 5, "config_name": "t5"}
#
#
# def solver_mcts_t1() -> Tuple[Solver.__class__, Dict[str, Any]]:
#     return MonteCarloTreeSearch, {"t_planning": 1, "config_name": "t1"}
#

# def solver_orca() -> Tuple[Solver.__class__, Dict[str, Any]]:
#     return ORCASolver, {"t_planning": 1, "config_name": "t1"}


###########################################################################
# Script ##################################################################
###########################################################################
if __name__ == '__main__':
    scenarios = load_functions_from_module(module="scenarios", prefix="scenario_")
    solvers = load_functions_from_module(module="evaluation", prefix="solver_")

    results = {}
    for (scenario_name, scenario_func), (solver_name, solver_func) in product(scenarios.items(), solvers.items()):
        name = f"{solver_name}:{scenario_name}"

        # Initialise environment environment and solver as described in functions.
        logging.warning(f"Evaluation: solver {solver_name} in scenario {scenario_name}")
        env, goal = scenario_func()
        solver_class, solver_kwargs = solver_func()
        solver = solver_class(env, goal=goal, verbose=2, multiprocessing=False, **solver_kwargs)

        # Solve posed problem, until goal has been reached.
        ego_trajectory_opt, ado_traj = solver.solve(time_steps=5, max_cpu_time=1.0)

        # Log and visualise results for later comparison.
        results[name] = evaluate_metrics(ego_trajectory=ego_trajectory_opt, ado_trajectories=ado_traj, env=env, goal=goal)
        logging.warning(f"Evaluation ==> {results[name]}")

    logging.warning("Evaluation results:")
    time.sleep(0.2)  # wait until loop has finished
    pprint(results)
