from typing import Tuple

import torch

from mantrap.agents.agent import Agent
from mantrap.constants import agent_speed_max, env_dt_default
from mantrap.utility.shaping import check_state
from mantrap.utility.utility import build_state_vector


class IntegratorDTAgent(Agent):
    def __init__(
        self, position: torch.Tensor, velocity: torch.Tensor = torch.zeros(2), history: torch.Tensor = None, **kwargs
    ):
        super(IntegratorDTAgent, self).__init__(position, velocity, history=history, **kwargs)

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float = env_dt_default) -> torch.Tensor:
        """
        .. math:: vel_{t+1} = action
        .. math:: pos_{t+1} = pos_t + vel_{t+1} * dt
        """
        assert check_state(state, enforce_temporal=False)
        assert action.size() == torch.Size([2])

        velocity_new = action.float()
        position_new = (state[0:2] + velocity_new * dt).float()
        return build_state_vector(position_new, velocity_new)

    def inverse_dynamics(self, state: torch.Tensor, state_previous: torch.Tensor, dt: float) -> torch.Tensor:
        """
        .. math:: action = (pos_t - pos_{t-1}) / dt
        """
        assert check_state(state, enforce_temporal=False)
        assert check_state(state_previous, enforce_temporal=False)

        return (state[0:2] - state_previous[0:2]) / dt

    def control_limits(self) -> Tuple[float, float]:
        """
        .. math:: [- v_{max}, v_{max}]
        """
        return -agent_speed_max, agent_speed_max
