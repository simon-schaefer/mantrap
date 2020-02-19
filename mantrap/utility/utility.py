import torch

from mantrap.utility.shaping import check_state, check_ego_trajectory


def build_state_vector(position: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """Stack position, orientation and velocity vector to build a full state vector, either np array or torch tensor."""
    assert type(position) == type(velocity)

    state = torch.cat((position, velocity))
    assert check_state(state, enforce_temporal=False), "constructed state vector has invalid shape"
    return state


def expand_state_vector(state_4: torch.Tensor, time: float) -> torch.Tensor:
    """Expand 4 dimensional (x, y, vx, vy) state vector by time information."""
    assert state_4.size() == torch.Size([4]), "timeless state vector is invalid"

    state = torch.cat((state_4, torch.ones(1) * time))
    assert check_state(state, enforce_temporal=True), "constructed state vector has invalid shape"
    return state


def build_trajectory_from_path(positions: torch.Tensor, dt: float, t_start: float = 0.0) -> torch.Tensor:
    """Derive (position, orientation, velocity)-trajectory information from position data only, assuming single
    integrator dynamics, i.e. v_i = (x_i+1 - x_i) / dt. """
    assert len(positions.shape) == 2, "primitives shape should be (t_horizon, 2)"
    assert positions.shape[1] == 2, "primitives shape should be (t_horizon, 2)"

    t_horizon = positions.shape[0]
    trajectory = torch.zeros((t_horizon, 5))

    trajectory[:, 0:2] = positions
    trajectory[1:, 2:4] = trajectory[1:, 0:2] - trajectory[:-1, 0:2]
    trajectory[0, 2:4] = trajectory[1, 2:4]  # otherwise first velocity always will be zero (!)
    trajectory[:, 4] = torch.linspace(t_start, t_start + t_horizon * dt, steps=t_horizon)

    assert check_ego_trajectory(trajectory, t_horizon=t_horizon)
    return trajectory
