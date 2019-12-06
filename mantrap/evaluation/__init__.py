import logging
from typing import Callable, Tuple

import numpy as np
import pandas as pd

from mantrap.simulation.abstract import Simulation
from mantrap.utility.io import path_from_home_directory

from .baselines import *
from .metrics import metrics
from .visualization import plot_scene


def evaluate(
    tag: str,
    ego_trajectory: np.ndarray,
    ado_trajectories: np.ndarray,
    sim: Simulation,
    goal: np.ndarray,
    baseline: Callable[[Simulation, np.ndarray, int], Tuple[np.ndarray, np.ndarray]],
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:

    eval_df, ado_traj_wo = metrics(sim, ego_trajectory=ego_trajectory, ado_trajectories=ado_trajectories)
    output_dir = path_from_home_directory(f"test/graphs/{tag}")
    plot_scene(
        ado_trajectories=ado_trajectories,
        ado_colors=sim.ado_colors,
        ado_ids=sim.ado_ids,
        ego_trajectory=ego_trajectory,
        ado_trajectories_wo=ado_traj_wo,
        output_dir=output_dir,
    )

    # Baseline evaluation and plotting.
    ego_traj_base, ado_traj_base = baselines.straight_line(sim=sim, goal=goal)
    eval_df_base, ado_traj_wo_base = metrics(sim, ego_trajectory=ego_traj_base, ado_trajectories=ado_traj_base)
    output_dir_base = path_from_home_directory(f"test/graphs/{tag}_{baseline.__name__}")
    plot_scene(
        ado_trajectories=ado_traj_base,
        ado_colors=sim.ado_colors,
        ado_ids=sim.ado_ids,
        ego_trajectory=ego_traj_base,
        ado_trajectories_wo=ado_traj_base,
        output_dir=output_dir_base,
    )

    # Plot scene without any ego interaction.
    output_dir_wo = path_from_home_directory(f"test/graphs/{tag}_wo")
    plot_scene(
        ado_trajectories=ado_traj_wo,
        ado_colors=sim.ado_colors,
        ado_ids=sim.ado_ids,
        ego_trajectory=None,
        ado_trajectories_wo=None,
        output_dir=output_dir_wo,
    )

    # Check whether actually the same "thing" has been compared by comparing the ado trajectories without
    # ego interaction (from metrics calculation).
    assert np.isclose(np.linalg.norm(ado_traj_wo - ado_traj_wo_base), 0, atol=0.1), "ado_wo trajectories do not match"
    logging.info(f"Metrics {tag} - solver:")
    print(eval_df)
    logging.info(f"Metrics {tag} - baseline {baseline.__name__}:")
    print(eval_df_base)
    return eval_df, eval_df_base, ado_traj_wo, ado_traj_base
