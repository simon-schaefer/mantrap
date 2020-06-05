import torch


def check_ego_state(x: torch.Tensor, enforce_temporal: bool = False) -> bool:
    assert not torch.any(torch.isnan(x))
    if enforce_temporal:
        assert x.size() == torch.Size([5])
    else:
        assert x.numel() in [4, 5]
    return True


def check_ego_action(x: torch.Tensor) -> bool:
    assert not torch.any(torch.isnan(x))
    assert x.size() == torch.Size([2])
    return True


def check_ego_path(x: torch.Tensor, t_horizon: int = None) -> bool:
    assert not torch.any(torch.isnan(x))
    assert len(x.shape) == 2  # (t_horizon, 2)
    assert x.shape[1] == 2
    if t_horizon is not None:
        assert x.shape[0] == t_horizon
    return True


def check_ego_controls(x: torch.Tensor, t_horizon: int = None) -> bool:
    assert not torch.any(torch.isnan(x))
    assert len(x.shape) == 2  # (t_horizon, dims)
    assert x.shape[-1] == 2
    if t_horizon is not None:
        assert x.shape[0] == t_horizon
    return True


def check_ego_trajectory(
    x: torch.Tensor,
    t_horizon: int = None,
    pos_only: bool = False,
    pos_and_vel_only: bool = False
) -> bool:
    assert not torch.any(torch.isnan(x))
    assert len(x.shape) == 2  # (t_horizon, 5)
    if pos_only:
        assert x.shape[1] >= 2  # (x, y, vx, vy)
    elif pos_and_vel_only:
        assert x.shape[1] >= 4  # (x, y, vx, vy)
    else:
        assert x.shape[1] == 5  # (x, y, vx, vy, t)
    if t_horizon is not None:
        assert x.shape[0] == t_horizon
    return True


def check_ado_states(x: torch.Tensor, num_ados: int = None, enforce_temporal: bool = False) -> bool:
    assert not torch.any(torch.isnan(x))
    assert len(x.shape) == 2  # (num_ados, 4/5)
    if num_ados is not None:
        assert x.shape[0] == num_ados
    if enforce_temporal:
        assert x.shape[1] == 5
    else:
        assert x.shape[1] in [4, 5]
    return True


def check_ado_history(x: torch.Tensor, ados: int = None) -> bool:
    assert not torch.any(torch.isnan(x))
    assert len(x.shape) == 3  # (num_ados, t_horizon, 2/4/5)
    assert x.shape[2] in [2, 4, 5]
    if ados is not None:
        assert x.shape[0] == ados
    return True


def check_ado_trajectories(x: torch.Tensor, t_horizon: int = None, ados: int = None, num_modes: int = None) -> bool:
    assert not torch.any(torch.isnan(x))
    assert len(x.shape) == 4  # (num_ados, t_horizon, num_modes, 2)

    assert x.shape[3] in [2, 5]  # (x, y) - positions, (x, y, vx, vy, t) - full state
    if ados is not None:
        assert x.shape[0] == ados
    if t_horizon is not None:
        assert x.shape[1] == t_horizon
    if num_modes is not None:
        assert x.shape[2] == num_modes
    return True


def check_ado_samples(x: torch.Tensor, t_horizon: int = None, ados: int = None, num_samples: int = None) -> bool:
    assert not torch.any(torch.isnan(x))
    assert len(x.shape) == 5  # (num_ados,num_samples,t_horizon, num_modes, 2/5)

    if num_samples is not None:
        assert x.shape[1] == num_samples
    assert all([check_ado_trajectories(x[:, i, :, :, :], t_horizon, ados, num_modes=1) for i in range(x.shape[1])])
    return True


def check_goal(x: torch.Tensor) -> bool:
    return check_2d_vector(x)


def check_2d_vector(x: torch.Tensor) -> bool:
    assert not torch.any(torch.isnan(x))
    assert x.size() == torch.Size([2])
    return True
