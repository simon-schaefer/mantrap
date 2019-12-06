from copy import deepcopy
from typing import Tuple

import numpy as np
import pandas as pd

from mantrap.simulation.abstract import Simulation
from mantrap.utility.shaping import check_ego_trajectory, check_ado_trajectories, extract_ado_trajectories


def metrics(
    sim: Simulation, ado_trajectories: np.ndarray, ego_trajectory: np.ndarray
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Evaluate goodness of ego trajectory by comparing the associated (predicted) ado trajectories with the
    (predicted) ado trajectories without any interaction with ego (i.e. no ego in the scene).
    Therefore forward simulate the scene without any ego to derive the ado trajectories without ego, then for
    comparison determine the accumulated L2 distance between both trajectories for each ado in the scene.
    :param sim: simulation environment.
    :param ado_trajectories: ado trajectories with ego interaction, (num_ados, num_modes=1, t_horizon, 5).
    :param ego_trajectory: ego trajectory to evaluate (t_horizon, 5).
    :return: accumulated L2 distance score
    """
    assert check_ado_trajectories(ado_trajectories=ado_trajectories, num_ados=sim.num_ados, num_modes=1)
    num_ados, num_modes, t_horizon = extract_ado_trajectories(ado_trajectories)
    assert check_ego_trajectory(ego_trajectory=ego_trajectory, t_horizon=t_horizon)

    # Reset simulation to initial states and determine ado trajectories without ego interaction by forward simulation.
    eval_sim = deepcopy(sim)  # intrinsically copy all simulation parameters such as dt (!)
    ado_trajectories_wo = eval_sim.predict(t_horizon=t_horizon, ego_trajectory=None)

    # Determine L2 norm of ado trajectories with and without ego.
    evaluation_df = pd.DataFrame()
    evaluation_df["ids"] = sim.ado_ids
    evaluation_df["position"] = [
        np.linalg.norm(ado_trajectories[i, 0, :, 0:2] - ado_trajectories_wo[i, 0, :, 0:2], ord=2)
        for i in range(num_ados)
    ]
    evaluation_df["velocity"] = [
        np.linalg.norm(ado_trajectories[i, 0, :, 3:5] - ado_trajectories_wo[i, 0, :, 3:5], ord=2)
        for i in range(num_ados)
    ]
    return evaluation_df, ado_trajectories_wo
