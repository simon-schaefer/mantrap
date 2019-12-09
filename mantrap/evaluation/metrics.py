from typing import Dict, Tuple

import numpy as np

from mantrap.simulation.abstract import Simulation
from mantrap.utility.shaping import check_ego_trajectory, check_ado_trajectories, extract_ado_trajectories


def metrics(
    sim: Simulation, ado_trajectories: np.ndarray, ego_trajectory: np.ndarray, ego_goal: np.ndarray
) -> Tuple[Dict, np.ndarray]:
    """Evaluate goodness of ego trajectory by comparing the associated (predicted) ado trajectories with the
    (predicted) ado trajectories without any interaction with ego (i.e. no ego in the scene).
    Therefore forward simulate the scene without any ego to derive the ado trajectories without ego, then for
    comparison determine the accumulated L2 distance between both trajectories for each ado in the scene.
    :param sim: simulation environment.
    :param ado_trajectories: ado trajectories with ego interaction, (num_ados, num_modes=1, t_horizon, 5).
    :param ego_trajectory: ego trajectory to evaluate (t_horizon, 5).
    :param ego_goal: goal state of ego at last time-step (first to entries should be position).
    :return: accumulated L2 distance score
    """
    assert check_ado_trajectories(ado_trajectories=ado_trajectories, num_ados=sim.num_ados, num_modes=1)
    num_ados, num_modes, t_horizon = extract_ado_trajectories(ado_trajectories)
    assert check_ego_trajectory(ego_trajectory=ego_trajectory, t_horizon=t_horizon)

    # Reset simulation to initial states and determine ado trajectories without ego interaction by forward simulation.
    ado_trajectories_wo = sim.predict(t_horizon=t_horizon, ego_trajectory=None)  # predict does not change class params

    # Determine L2 norm of ado trajectories with and without ego.
    evaluation_dict = {"ados": {}, "ego": {}}
    for i, ado in enumerate(sim.ados):
        evaluation_dict["ados"][ado.id] = {
            "position_dev": np.linalg.norm(ado_trajectories[i, 0, :, 0:2] - ado_trajectories_wo[i, 0, :, 0:2], ord=2),
            "velocity_dev": np.linalg.norm(ado_trajectories[i, 0, :, 3:5] - ado_trajectories_wo[i, 0, :, 3:5], ord=2),
        }
    evaluation_dict["ego"]["final_distance"] = np.linalg.norm(ego_trajectory[-1, 0:2] - ego_goal[0:2])
    return evaluation_dict, ado_trajectories_wo
