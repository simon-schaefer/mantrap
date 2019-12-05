from typing import Tuple

import numpy as np


def check_ego_trajectory(ego_trajectory: np.ndarray) -> bool:
    is_correct = True
    is_correct = is_correct and len(ego_trajectory) == 2  # (t_horizon, 6)
    is_correct = is_correct and ego_trajectory.shape[1] == 6  # (x, y, theta, vx, vy, t)
    return is_correct


def check_ado_trajectories(ado_trajectories: np.ndarray) -> bool:
    is_correct = True
    is_correct = is_correct and len(ado_trajectories.shape) == 4  # (num_ados,num_samples,t_horizon,6)
    is_correct = is_correct and ado_trajectories.shape[3] == 6  # (x, y, theta, vx, vy, t)
    return is_correct


def extract_ado_trajectories(ado_trajectories: np.ndarray) -> Tuple[int, int, int]:
    assert check_ado_trajectories(ado_trajectories=ado_trajectories)
    num_ados = ado_trajectories.shape[0]
    num_modes = ado_trajectories.shape[1]
    t_horizon = ado_trajectories.shape[2]
    return num_ados, num_modes, t_horizon
