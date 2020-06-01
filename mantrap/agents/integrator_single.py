import math
import typing

import torch

from .base.linear import LinearDTAgent


class IntegratorDTAgent(LinearDTAgent):
    """Linear single integrator dynamics:

    .. math:: vel_{t+1} = action
    .. math:: pos_{t+1} = pos_t + vel_{t+1} * dt
    """

    @staticmethod
    def _dynamics_matrices(dt: float, x: torch.Tensor = None
                           ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        A = torch.tensor([[1, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1]])
        B = torch.tensor([[dt, 0, 1, 0, 0],
                          [0, dt, 0, 1, 0]]).t()
        T = torch.tensor([0, 0, 0, 0, dt]).t()
        return A, B, T

    @staticmethod
    def dynamics_scalar(px: float, py: float, vx: float, vy: float, ux: float, uy: float, dt: float
                        ) -> typing.Tuple[float, float, float, float]:
        return px + ux * dt, py + uy * dt, ux, uy

    def _inverse_dynamics(self, state: torch.Tensor, state_previous: torch.Tensor, dt: float) -> torch.Tensor:
        """
        .. math:: action = (pos_t - pos_{t-1}) / dt
        """
        return (state[0:2] - state_previous[0:2]) / dt

    def _inverse_dynamics_batch(self, batch: torch.Tensor, dt: float) -> torch.Tensor:
        return (batch[1:, 0:2] - batch[:-1, 0:2]) / dt

    ###########################################################################
    # Agent control functions #################################################
    ###########################################################################
    def control_limits(self) -> typing.Tuple[float, float]:
        """
        .. math:: [- v_{max}, v_{max}]
        """
        return self.speed_limits

    ###########################################################################
    # Agent Properties ########################################################
    ###########################################################################
    @property
    def acceleration_max(self) -> float:
        """Since single integrators are assumed to react instantly, their maximal acceleration
        is in fact infinite !"""
        return math.inf

    @staticmethod
    def agent_type() -> str:
        return "single_int"
