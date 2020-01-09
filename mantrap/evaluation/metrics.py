from typing import Dict, List, Tuple

import numpy as np

from mantrap.utility.shaping import check_ego_trajectory, check_ado_trajectories, extract_ado_trajectories


def metrics(
    ado_trajs: np.ndarray,
    ado_trajs_wo: np.ndarray,
    ado_ids: List[str],
    ego_trajectory: np.ndarray,
    ego_goal: np.ndarray,
) -> Tuple[Dict, np.ndarray]:
    """Evaluate goodness of ego trajectory by comparing the associated (predicted) ado trajectories with the
    (predicted) ado trajectories without any interaction with ego (i.e. no ego in the scene).
    Therefore forward simulate the scene without any ego to derive the ado trajectories without ego, then for
    comparison determine the accumulated L2 distance between both trajectories for each ado in the scene.
    :param ado_trajs: ado trajectories with ego interaction, (num_ados, num_modes=1, t_horizon, 5).
    :param ado_trajs_wo: ado trajectories without ego interaction by forward simulation, shape == ado_trajs.shape.
    :param ado_ids: ado identifier strings.
    :param ego_trajectory: ego trajectory to evaluate (t_horizon, 5).
    :param ego_goal: goal state of ego at last time-step (first to entries should be position).
    :return: accumulated L2 distance score
    """
    num_ados = ado_trajs.shape[0]

    assert check_ado_trajectories(ado_trajectories=ado_trajs, num_ados=num_ados, num_modes=1)
    assert check_ado_trajectories(ado_trajectories=ado_trajs_wo, num_ados=num_ados, num_modes=1)
    num_ados, num_modes, t_horizon = extract_ado_trajectories(ado_trajs)
    assert check_ego_trajectory(ego_trajectory=ego_trajectory, t_horizon=t_horizon)

    # Determine L2 norm of ado trajectories with and without ego.
    evaluation_dict = {"ados": {}, "ego": {}}
    for i, ado_id in enumerate(ado_ids):
        evaluation_dict["ados"][ado_id] = {
            "position_dev": np.linalg.norm(ado_trajs[i, 0, :, 0:2] - ado_trajs_wo[i, 0, :, 0:2], ord=2),
            "velocity_dev": np.linalg.norm(ado_trajs[i, 0, :, 3:5] - ado_trajs_wo[i, 0, :, 3:5], ord=2),
        }
    evaluation_dict["ego"]["final_distance"] = np.linalg.norm(ego_trajectory[-1, 0:2] - ego_goal[0:2])
    evaluation_dict["ego"]["traj_length"] = ego_trajectory.shape[0]
    return evaluation_dict, ado_trajs_wo
