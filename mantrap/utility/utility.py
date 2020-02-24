import torch

from mantrap.utility.shaping import check_state


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
