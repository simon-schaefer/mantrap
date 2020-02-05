import torch

from mantrap.utility.shaping import check_state, check_ego_trajectory


def build_state_vector(position: torch.Tensor, theta: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """Stack position, orientation and velocity vector to build a full state vector, either np array or torch tensor."""
    assert type(position) == type(velocity) == type(theta)

    state = torch.cat((position, theta, velocity))
    assert check_state(state, enforce_temporal=False), "constructed state vector has invalid shape"
    return state


def expand_state_vector(state_5: torch.Tensor, time: float) -> torch.Tensor:
    """Expand 5 dimensional (x, y, theta, vx, vy) state vector by time information."""
    assert state_5.size() == torch.Size([5]), "timeless state vector is invalid"

    state = torch.cat((state_5, torch.ones(1) * time))
    assert check_state(state, enforce_temporal=True), "constructed state vector has invalid shape"
    return state


def build_trajectory_from_positions(positions: torch.Tensor, dt: float, t_start: float = 0.0) -> torch.Tensor:
    assert len(positions.shape) == 2, "primitives shape should be (t_horizon, 2)"
    assert positions.shape[1] == 2, "primitives shape should be (t_horizon, 2)"

    t_horizon = positions.shape[0]
    trajectory = torch.zeros((t_horizon, 6))

    trajectory[:, 0:2] = positions.float()
    trajectory[1:, 3:5] = trajectory[1:, 0:2] - trajectory[:-1, 0:2]
    trajectory[:, 2] = torch.atan2(trajectory[:, 4], trajectory[:, 3])
    trajectory[:, 5] = torch.linspace(t_start, t_start + t_horizon * dt, steps=t_horizon)

    assert check_ego_trajectory(trajectory, t_horizon=t_horizon)
    return trajectory
