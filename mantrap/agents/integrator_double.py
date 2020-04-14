from typing import Tuple

import torch

from mantrap.agents.agent import Agent
from mantrap.constants import AGENT_ACC_MAX, ENV_DT_DEFAULT
from mantrap.utility.shaping import check_ego_action, check_ego_state


class DoubleIntegratorDTAgent(Agent):
    def __init__(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor = torch.zeros(2),
        history: torch.Tensor = None, **kwargs
    ):
        super(DoubleIntegratorDTAgent, self).__init__(position, velocity, history=history, **kwargs)

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float = ENV_DT_DEFAULT) -> torch.Tensor:
        """
          .. math:: vel_{t+1} = vel_t + action * dt
          .. math:: pos_{t+1} = pos_t + vel_{t+1} * dt + 0.5 * action * dt^2
          """
        assert check_ego_state(state, enforce_temporal=False)  # (x, y, theta, vx, vy)
        assert action.size() == torch.Size([2])  # (vx, vy)
        action = action.float()

        velocity_new = (state[2:4] + action * dt).float()
        position_new = (state[0:2] + state[2:4] * dt + 0.5 * action * dt ** 2).float()
        return self.build_state_vector(position_new, velocity_new)

    def inverse_dynamics(self, state: torch.Tensor, state_previous: torch.Tensor, dt: float) -> torch.Tensor:
        """
        .. math:: action = (vel_t - vel_{t-1}) / dt
        """
        assert check_ego_state(state, enforce_temporal=False)
        assert check_ego_state(state_previous, enforce_temporal=False)

        action = (state[2:4] - state_previous[2:4]) / dt
        assert check_ego_action(x=action)
        return action

    def control_limits(self) -> Tuple[float, float]:
        """
        .. math:: [- a_{max}, a_{max}]
        """
        return -AGENT_ACC_MAX, AGENT_ACC_MAX
