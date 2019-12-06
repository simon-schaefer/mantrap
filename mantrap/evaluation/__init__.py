import logging
import os
from typing import Callable, Tuple

import numpy as np
import pandas as pd

from mantrap.constants import planning_horizon_default
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
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:

    eval_df, ado_traj_wo = metrics(sim, ego_trajectory=ego_trajectory, ado_trajectories=ado_trajectories)

    # Baseline evaluation and plotting.
    ego_traj_base, ado_traj_base = baseline(sim, goal, planning_horizon_default)
    eval_df_base, ado_traj_wo_base = metrics(sim, ego_trajectory=ego_traj_base, ado_trajectories=ado_traj_base)

    # Check whether actually the same "thing" has been compared by comparing the ado trajectories without
    # ego interaction (from metrics calculation).
    assert np.isclose(np.linalg.norm(ado_traj_wo - ado_traj_wo_base), 0, atol=0.1), "ado_wo trajectories do not match"
    logging.info(f"Metrics {tag} - solver:")
    print(eval_df)
    logging.info(f"Metrics {tag} - baseline {baseline.__name__}:")
    print(eval_df_base)

    # Visualization.
    if do_visualization:
        import matplotlib.pyplot as plt
        output_dir = path_from_home_directory(f"test/graphs/{tag}")
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('seaborn')
        for t in range(ego_trajectory.shape[0]):
            fig = plt.figure(constrained_layout=True)
            gs = plt.GridSpec(3, 2, wspace=0.2, hspace=0.4, figure=fig)

            ax1 = fig.add_subplot(gs[0:2, :])
            ax1 = plot_scene(
                ax=ax1, t=t,
                ado_trajectories=ado_trajectories,
                ado_colors=sim.ado_colors,
                ado_ids=sim.ado_ids,
                ego_trajectory=ego_trajectory,
                ado_trajectories_wo=ado_traj_wo,
            )

            ax2 = fig.add_subplot(gs[2, 0])
            ax2 = plot_scene(
                ax=ax2, t=t,
                ado_trajectories=ado_traj_wo,
                ado_colors=sim.ado_colors,
                ado_ids=sim.ado_ids,
                ego_trajectory=None,
                ado_trajectories_wo=None,
            )
            ax2.set_title("no ego agent")

            ax3 = fig.add_subplot(gs[2, 1])
            ax3 = plot_scene(
                ax=ax3, t=t,
                ado_trajectories=ado_traj_base,
                ado_colors=sim.ado_colors,
                ado_ids=sim.ado_ids,
                ego_trajectory=ego_traj_base,
                ado_trajectories_wo=ado_traj_base,
            )
            ax3.set_title(f"baseline: {baseline.__name__}")

            fig.suptitle(f"task: {tag}")
            plt.savefig(os.path.join(output_dir, f"{t:04d}.png"))
            plt.close()

    return eval_df, eval_df_base, ado_traj_wo, ado_traj_base
