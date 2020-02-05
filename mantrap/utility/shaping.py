from typing import Tuple

import torch


def check_state(state: torch.Tensor, enforce_temporal: bool = False) -> bool:
    if enforce_temporal:
        is_correct = state.size() == torch.Size([6])
    else:
        is_correct = state.size() == torch.Size([5]) or state.size() == torch.Size([6])
    return is_correct


def check_policies(policies: torch.Tensor, t_horizon: int = None, num_ados: int = None, num_modes: int = None):
    is_correct = True
    is_correct = is_correct and len(policies.shape) == 4  # (num_ados, num_modes, dims)
    if t_horizon is not None:
        is_correct = is_correct and policies.shape[2] == t_horizon
    if num_ados is not None:
        is_correct = is_correct and policies.shape[0] == num_ados
    if num_modes is not None:
        is_correct = is_correct and policies.shape[1] == num_modes
    return is_correct


def check_weights(weights: torch.Tensor, num_ados: int = None, num_modes: int = None) -> bool:
    is_correct = True
    is_correct = is_correct and len(weights.shape) == 2  # (num_ados, num_modes)
    if num_ados is not None:
        is_correct = is_correct and weights.shape[0] == num_ados
    if num_modes is not None:
        is_correct = is_correct and weights.shape[1] == num_modes
    return is_correct


def check_trajectory_primitives(primitives: torch.Tensor, num_primitives: int = None, t_horizon: int = None) -> bool:
    is_correct = True
    if num_primitives is not None:
        is_correct = is_correct and len(primitives.shape) == 3  # (num_primitives, t_horizon, 2)
        is_correct = is_correct and primitives.shape[2] == 2
        is_correct = is_correct and primitives.shape[0] == num_primitives
        if t_horizon is not None:
            is_correct = is_correct and primitives.shape[1] == t_horizon
    else:
        is_correct = is_correct and len(primitives.shape) == 2  # (t_horizon, 2)
        if t_horizon is not None:
            is_correct = is_correct and primitives.shape[0] == t_horizon
    return is_correct


def check_ego_trajectory(ego_trajectory: torch.Tensor, t_horizon: int = None) -> bool:
    is_correct = True
    is_correct = is_correct and len(ego_trajectory.shape) == 2  # (t_horizon, 6)
    is_correct = is_correct and ego_trajectory.shape[1] == 6  # (x, y, theta, vx, vy, t)
    if t_horizon is not None:
        is_correct = is_correct and ego_trajectory.shape[0] == t_horizon
    return is_correct


def check_trajectories(trajectories: torch.Tensor, ados: int = None, modes: int = None, t_horizon: int = None) -> bool:
    is_correct = True
    is_correct = is_correct and len(trajectories.shape) == 4  # (num_ados,num_modes,t_horizon,6)
    is_correct = is_correct and trajectories.shape[3] == 6  # (x, y, theta, vx, vy, t)
    if ados is not None:
        is_correct = is_correct and trajectories.shape[0] == ados
    if modes is not None:
        is_correct = is_correct and trajectories.shape[1] == modes
    if t_horizon is not None:
        is_correct = is_correct and trajectories.shape[2] == t_horizon
    return is_correct


def extract_ado_trajectories(ado_trajectories: torch.Tensor) -> Tuple[int, int, int]:
    num_ados = ado_trajectories.shape[0]
    num_modes = ado_trajectories.shape[1]
    t_horizon = ado_trajectories.shape[2]
    return num_ados, num_modes, t_horizon
