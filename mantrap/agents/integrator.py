import torch

from mantrap.agents.agent import Agent
from mantrap.constants import sim_dt_default
from mantrap.utility.shaping import check_state
from mantrap.utility.utility import build_state_vector


class IntegratorDTAgent(Agent):
    def __init__(
        self, position: torch.Tensor, velocity: torch.Tensor = torch.zeros(2), history: torch.Tensor = None, **kwargs
    ):
        super(IntegratorDTAgent, self).__init__(position, velocity, history=history, **kwargs)

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float = sim_dt_default) -> torch.Tensor:
        assert check_state(state, enforce_temporal=False), "state should be two-dimensional (x, y, theta, vx, vy)"
        assert action.size() == torch.Size([2]), "action must be two-dimensional (vx, vy)"

        velocity_new = action.float()
        theta_new = torch.tensor([torch.atan2(velocity_new[1], velocity_new[0])])
        position_new = state[0:2] + velocity_new * dt
        return build_state_vector(position_new, theta_new, velocity_new)
