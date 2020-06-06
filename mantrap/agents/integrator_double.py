import typing

import torch

from .base.linear import LinearDTAgent


class DoubleIntegratorDTAgent(LinearDTAgent):
    """ Linear double integrator dynamics:

    .. math:: vel_{t+1} = vel_t + action * dt
    .. math:: pos_{t+1} = pos_t + vel_{t+1} * dt
    """

    @staticmethod
    def _dynamics_matrices(dt: float, x: torch.Tensor = None
                           ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        A = torch.tensor([[1, 0, dt, 0, 0],
                          [0, 1, 0, dt, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 1]])
        B = torch.tensor([[0, 0, dt, 0, 0],
                          [0, 0, 0, dt, 0]]).t()
        T = torch.tensor([0, 0, 0, 0, dt]).t()
        return A, B, T

    def _inverse_dynamics(self, state: torch.Tensor, state_previous: torch.Tensor, dt: float) -> torch.Tensor:
        """
        .. math:: action = (vel_t - vel_{t-1}) / dt
        """
        return (state[2:4] - state_previous[2:4]) / dt

    def _inverse_dynamics_batch(self, batch: torch.Tensor, dt: float) -> torch.Tensor:
        return (batch[1:, 2:4] - batch[:-1, 2:4]) / dt

    ###########################################################################
    # Agent control functions #################################################
    ###########################################################################
    def control_limits(self) -> typing.Tuple[float, float]:
        """
        .. math:: [- a_{max}, a_{max}]
        """
        return -self.acceleration_max, self.acceleration_max

    def control_norm(self, controls: torch.Tensor) -> torch.Tensor:
        """Compute the agent's control norm ||u|| = L1-norm.

        :param controls: controls to calculate the norm from (N, 2).
        :returns: control norm (N, 2).
        """
        return torch.abs(controls).float()

    ###########################################################################
    # Agent Properties ########################################################
    ###########################################################################
    @staticmethod
    def agent_type() -> str:
        return "double_int"
