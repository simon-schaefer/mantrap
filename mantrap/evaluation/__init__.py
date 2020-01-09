from pprint import pprint
import logging
import inspect
import os
from typing import Callable, Dict, Tuple

import numpy as np

from mantrap.agents.agent import Agent
from mantrap.evaluation import scenarios as evaluation_scenarios
from mantrap.simulation.simulation import Simulation
from mantrap.utility.io import path_from_home_directory
from mantrap.utility.shaping import check_ado_trajectories

from .metrics import metrics
from .visualization import plot_scene


def evaluate(
    tag: str,
    ego_trajectory: np.ndarray,
    ado_trajectories: np.ndarray,
    sim: Simulation,
    goal: np.ndarray,
    do_visualization: bool = True,
) -> Tuple[Dict, np.ndarray]:

    t_horizon = ego_trajectory.shape[0]
    assert check_ado_trajectories(ado_trajectories, num_ados=sim.num_ados, num_modes=sim.num_ado_modes)
    assert ado_trajectories.shape[2] == t_horizon

    ado_trajectories_wo = sim.predict(t_horizon=t_horizon, ego_trajectory=None)

    eval_dict, ado_traj_wo = metrics(
        ado_trajectories,
        ado_trajs_wo=ado_trajectories_wo,
        ado_ids=sim.ado_ids,
        ego_trajectory=ego_trajectory,
        ego_goal=goal,
    )

    # Check whether actually the same "thing" has been compared by comparing the ado trajectories without
    # ego interaction (from metrics calculation).
    logging.warning(f"Metrics on task {tag} -> solver:")
    pprint(eval_dict)

    # Visualization.
    if do_visualization:
        import matplotlib.pyplot as plt

        output_dir = path_from_home_directory(f"test/graphs/{tag}")
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use("seaborn")
        for t in range(ego_trajectory.shape[0]):
            fig = plt.figure(constrained_layout=True)
            gs = plt.GridSpec(3, 2, wspace=0.2, hspace=0.4, figure=fig)

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

            ax4 = fig.add_subplot(gs[2, 1])
            speed_ego = np.sqrt(ego_trajectory[:t, 3] ** 2 + ego_trajectory[:t, 4] ** 2)
            ax4.plot(ego_trajectory[:t, -1], speed_ego, label="solver")
            ax4.legend()

            fig.suptitle(f"task: {tag}")
            plt.savefig(os.path.join(output_dir, f"{t:04d}.png"))
            plt.close()
    print("\n")
    return eval_dict, ado_traj_wo


def eval_scenarios() -> Dict[str, Callable[[Simulation.__class__, Agent.__class__], Tuple[Simulation, np.ndarray]]]:
    scenario_functions = {}
    functions = [o for o in inspect.getmembers(evaluation_scenarios) if inspect.isfunction(o[1])]
    for function_tuple in functions:
        function_name, _ = function_tuple
        if function_name.startswith("scenario"):
            tag = function_name.replace("scenario_", "")
            scenario_functions[tag] = function_tuple[1]
    return scenario_functions
