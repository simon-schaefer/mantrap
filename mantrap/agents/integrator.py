from typing import Tuple

import torch

from mantrap.agents.agent import Agent
from mantrap.constants import AGENT_SPEED_MAX, ENV_DT_DEFAULT
from mantrap.utility.shaping import check_ego_action, check_ego_state

class IntegratorDTAgent(Agent):
    def __init__(
        self, position: torch.Tensor, velocity: torch.Tensor = torch.zeros(2), history: torch.Tensor = None, **kwargs
    ):
        super(IntegratorDTAgent, self).__init__(position, velocity, history=history, **kwargs)

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float = ENV_DT_DEFAULT) -> torch.Tensor:
        """
        .. math:: vel_{t+1} = action
        .. math:: pos_{t+1} = pos_t + vel_{t+1} * dt
        """
        assert check_ego_state(state, enforce_temporal=False)
        assert action.size() == torch.Size([2])

        velocity_new = action.float()
        position_new = (state[0:2] + velocity_new * dt).float()
        return self.build_state_vector(position_new, velocity_new)

    def inverse_dynamics(self, state: torch.Tensor, state_previous: torch.Tensor, dt: float) -> torch.Tensor:
        """
        .. math:: action = (pos_t - pos_{t-1}) / dt
        """
        assert check_ego_state(state, enforce_temporal=False)
        assert check_ego_state(state_previous, enforce_temporal=False)

        action = (state[0:2] - state_previous[0:2]) / dt
        assert check_ego_action(x=action)
        return action

    def control_limits(self) -> Tuple[float, float]:
        """
        .. math:: [- v_{max}, v_{max}]
        """
        return -AGENT_SPEED_MAX, AGENT_SPEED_MAX
