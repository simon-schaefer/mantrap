from typing import Tuple

import numpy as np


def check_state(state: np.ndarray, enforce_temporal: bool = False) -> bool:
    return state.size == 6 if enforce_temporal else (state.size == 5 or state.size == 6)


def check_policies(policies: np.ndarray, t_horizon: int = None, num_ados: int = None, num_modes: int = None):
    is_correct = True
    is_correct = is_correct and len(policies.shape) == 4  # (num_ados, num_modes, dims)
    if t_horizon is not None:
        is_correct = is_correct and policies.shape[2] == t_horizon
    if num_ados is not None:
        is_correct = is_correct and policies.shape[0] == num_ados
    if num_modes is not None:
        is_correct = is_correct and policies.shape[1] == num_modes
    return is_correct


def check_weights(weights: np.ndarray, num_ados: int = None, num_modes: int = None) -> bool:
    is_correct = True
    is_correct = is_correct and len(weights.shape) == 2  # (num_ados, num_modes)
    if num_ados is not None:
        is_correct = is_correct and weights.shape[0] == num_ados
    if num_modes is not None:
        is_correct = is_correct and weights.shape[1] == num_modes
    return is_correct


def check_ego_trajectory(ego_trajectory: np.ndarray, t_horizon: int = None) -> bool:
    is_correct = True
    is_correct = is_correct and len(ego_trajectory.shape) == 2  # (t_horizon, 6)
    is_correct = is_correct and ego_trajectory.shape[1] == 6  # (x, y, theta, vx, vy, t)
    if t_horizon is not None:
        is_correct = is_correct and ego_trajectory.shape[0] == t_horizon
    return is_correct


def check_ado_trajectories(
    trajectories: np.ndarray, num_ados: int = None, num_modes: int = None, t_horizon: int = None
) -> bool:
    is_correct = True
    is_correct = is_correct and len(trajectories.shape) == 4  # (num_ados,num_modes,t_horizon,6)
    is_correct = is_correct and trajectories.shape[3] == 6  # (x, y, theta, vx, vy, t)
    if num_ados is not None:
        is_correct = is_correct and trajectories.shape[0] == num_ados
    if num_modes is not None:
        is_correct = is_correct and trajectories.shape[1] == num_modes
    if t_horizon is not None:
        is_correct = is_correct and trajectories.shape[2] == t_horizon
    return is_correct


def extract_ado_trajectories(ado_trajectories: np.ndarray) -> Tuple[int, int, int]:
    num_ados = ado_trajectories.shape[0]
    num_modes = ado_trajectories.shape[1]
    t_horizon = ado_trajectories.shape[2]
    return num_ados, num_modes, t_horizon
