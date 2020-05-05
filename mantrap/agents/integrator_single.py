import math
import typing

import torch

from .base.linear import LinearDTAgent


class IntegratorDTAgent(LinearDTAgent):
    """Linear single integrator dynamics:

    .. math:: vel_{t+1} = action
    .. math:: pos_{t+1} = pos_t + vel_{t+1} * dt
    """

    def _dynamics_matrices(self, dt: float) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    def go_to_point(
        self,
        state: typing.Tuple[float, float, float, float],
        target_point: typing.Tuple[float, float],
        speed: float,
        dt: float,
    ) -> typing.Tuple[typing.Tuple[float, float, float, float], typing.Tuple[float, float]]:
        """Determine and execute the controls for going for the given state to some target point with respect to
        the internal dynamics.

        Due to the instant dynamics of a single integrator going from one point to another simple means instantly
        adapting the controls to go in the right direction.

        :param state: pos_and_vel_only state vector i.e. (px, py, vx, vy) for starting state.
        :param target_point: 2D target point (px, py).
        :param speed: preferable speed for state update [m/s].
        :param dt: update time interval [s].
        :returns: updated state at t = t0 + dt and used (cardinal) control input.
        """
        px, py, vx, vy = state
        target_point_x, target_point_y = target_point
        target_distance = math.hypot(px - target_point_x, py - target_point_y)
        if math.isclose(target_distance, 0.0):
            return (px, py, vx, vy), (0.0, 0.0)

        # Determine cartesian controls with "infinite" speed. Then adapt the controls to the given reference
        # speed value (while keeping the direction).
        ux, uy = target_point_x - px, target_point_y - py
        u_speed = math.hypot(ux, uy)
        ux, uy = ux / u_speed * speed, uy / u_speed * speed

        # Ensure feasibility of control input given internal control limits.
        ux, uy = self.make_controls_feasible_scalar(ux, uy)

        # Do simulation step, i.e. update agent with computed control input.
        px, py, vx, vy = self.dynamics_scalar(px, py, vx, vy, ux, uy, dt)
        return (px, py, vx, vy), (ux, uy)

    def control_limits(self) -> typing.Tuple[float, float]:
        """
        .. math:: [- v_{max}, v_{max}]
        """
        return -self.speed_max, self.speed_max

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
