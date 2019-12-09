from pprint import pprint
import logging
import inspect
import os
from typing import Callable, Dict, Tuple

import numpy as np

from mantrap.agents.agent import Agent
from mantrap.constants import planning_horizon_default
from mantrap.evaluation import scenarios as evaluation_scenarios
from mantrap.simulation.abstract import Simulation
from mantrap.utility.io import path_from_home_directory

from .metrics import metrics
from .visualization import plot_scene


def evaluate(
    tag: str,
    ego_trajectory: np.ndarray,
    ado_trajectories: np.ndarray,
    sim: Simulation,
    goal: np.ndarray,
    baseline: Callable[[Simulation, np.ndarray, int], Tuple[np.ndarray, np.ndarray]],
    do_visualization: bool = True,
) -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:

    eval_dict, ado_traj_wo = metrics(sim, ado_trajectories, ego_trajectory=ego_trajectory, ego_goal=goal)

    # Baseline evaluation and plotting.
    ego_traj_base, ado_traj_base = baseline(sim, goal, planning_horizon_default)
    eval_dict_base, ado_traj_wo_base = metrics(sim, ado_traj_base, ego_trajectory=ego_traj_base, ego_goal=goal)

    # Check whether actually the same "thing" has been compared by comparing the ado trajectories without
    # ego interaction (from metrics calculation).
    assert np.isclose(np.linalg.norm(ado_traj_wo - ado_traj_wo_base), 0, atol=0.1), "ado_wo trajectories do not match"
    logging.info(f"Metrics on task {tag} -> solver:")
    pprint(eval_dict)
    logging.info(f"Metrics on task {tag} -> baseline {baseline.__name__}:")
    pprint(eval_dict_base)

    # Visualization.
    if do_visualization:
        import matplotlib.pyplot as plt

        output_dir = path_from_home_directory(f"test/graphs/{tag}")
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use("seaborn")
        for t in range(ego_trajectory.shape[0]):
            fig = plt.figure(constrained_layout=True)
            gs = plt.GridSpec(4, 2, wspace=0.2, hspace=0.4, figure=fig)

            ax1 = fig.add_subplot(gs[0:2, :])
            ax1 = plot_scene(
                ax=ax1,
                t=t,
                ado_trajectories=ado_trajectories,
                ado_colors=sim.ado_colors,
                ado_ids=sim.ado_ids,
                ego_trajectory=ego_trajectory,
                ado_trajectories_wo=ado_traj_wo,
            )

            ax2 = fig.add_subplot(gs[2, 0])
            ax2 = plot_scene(
                ax=ax2,
                t=t,
                ado_trajectories=ado_traj_wo,
                ado_colors=sim.ado_colors,
                ado_ids=sim.ado_ids,
                ego_trajectory=None,
                ado_trajectories_wo=None,
            )
            ax2.set_title("no ego agent")

            ax3 = fig.add_subplot(gs[2, 1])
            ax3 = plot_scene(
                ax=ax3,
                t=t,
                ado_trajectories=ado_traj_base,
                ado_colors=sim.ado_colors,
                ado_ids=sim.ado_ids,
                ego_trajectory=ego_traj_base,
                ado_trajectories_wo=ado_traj_wo_base,
            )
            ax3.set_title(f"baseline: {baseline.__name__}")

            ax4 = fig.add_subplot(gs[3, :])
            speed_ego = np.sqrt(ego_trajectory[:t, 3] ** 2 + ego_trajectory[:t, 4] ** 2)
            speed_ego_base = np.sqrt(ego_traj_base[:t, 3] ** 2 + ego_traj_base[:t, 4] ** 2)
            ax4.plot(ego_trajectory[:t, -1], speed_ego, label="solver")
            ax4.plot(ego_trajectory[:t, -1], speed_ego_base, label="baseline")
            ax4.legend()

            fig.suptitle(f"task: {tag}")
            plt.savefig(os.path.join(output_dir, f"{t:04d}.png"))
            plt.close()
    print("\n")
    return eval_dict, eval_dict_base, ado_traj_wo, ado_traj_base


def eval_scenarios() -> Dict[str, Callable[[Agent.__class__], Tuple[Simulation, np.ndarray]]]:
    scenario_functions = {}
    functions = [o for o in inspect.getmembers(evaluation_scenarios) if inspect.isfunction(o[1])]
    for function_tuple in functions:
        function_name, _ = function_tuple
        if function_name.startswith("scenario"):
            tag = function_name.replace("scenario_", "")
            scenario_functions[tag] = function_tuple[1]
    return scenario_functions
