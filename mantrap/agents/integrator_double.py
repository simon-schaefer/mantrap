import math
from typing import Tuple

import torch

from mantrap.agents.agent_intermediates.linear import LinearAgent
from mantrap.utility.maths import Circle


class DoubleIntegratorDTAgent(LinearAgent):
    """ Linear double integrator dynamics:

    .. math:: vel_{t+1} = vel_t + action * dt
    .. math:: pos_{t+1} = pos_t + vel_{t+1} * dt
    """

    def _dynamics_matrices(self, dt: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        A = torch.tensor([[1, 0, dt, 0, 0],
                          [0, 1, 0, dt, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 1]])
        B = torch.tensor([[0, 0, dt, 0, 0],
                          [0, 0, 0, dt, 0]]).t()
        T = torch.tensor([0, 0, 0, 0, dt]).t()
        return A, B, T

    @staticmethod
    def dynamics_scalar(px: float, py: float, vx: float, vy: float, ux: float, uy: float, dt: float
                        ) -> Tuple[float, float, float, float]:
        return px + vx * dt, py + vy * dt, vx + ux * dt, vy + uy * dt

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
    def go_to_point(
        self,
        state: Tuple[float, float, float, float],
        target_point: Tuple[float, float],
        speed: float,
        dt: float,
        pseudo_wheel_distance: float = 0.05,
        k_speed: float = 1.0,
    ) -> Tuple[Tuple[float, float, float, float], Tuple[float, float]]:
        """Determine and execute the controls for going for the given state to some target point with respect to
        the internal dynamics.

        For from some point to another the double integrator uses a variant of pure pursuit control, which basically
        means, that it is not treated as a point mass but as a vehicle/car with very small width
        (`pseudo_wheel_distance`). Then the required acceleration and steering commands can be computed for the
        given scenario and transformed to the cartesian controls the double integrator requires.

        :param state: pos_and_vel_only state vector i.e. (px, py, vx, vy) for starting state.
        :param target_point: 2D target point (px, py).
        :param speed: preferable speed for state update [m/s].
        :param dt: update time interval [s].
        :param pseudo_wheel_distance: distance between imaginary wheels [m].
        :param k_speed: speed control gain.
        :returns: updated state at t = t0 + dt and used (cardinal) control input.
        """
        px, py, vx, vy = state
        target_point_x, target_point_y = target_point
        target_distance = math.hypot(px - target_point_x, py - target_point_y)

        # Determine pseudo-vehicle state (speed and orientation).
        v = math.hypot(vx, vy)
        yaw = math.atan2(vy, vx)

        # Compute pseudo-vehicle controls using pure pursuit equations.
        d_vel = k_speed * (speed - v)
        alpha = math.atan2(target_point_y - py, target_point_x - px) - yaw
        delta = math.atan2(2.0 * pseudo_wheel_distance * math.sin(alpha) / target_distance, 1.0)
        d_yaw = v / pseudo_wheel_distance * math.tan(delta)

        # Transform pure pursuit control commands to control commands fitting the agent type.
        ux = 0.5 * d_vel * vx / v - vy * d_yaw
        uy = 0.5 * d_vel * vy / v + vx * d_yaw

        # Ensure feasibility of control input given internal control limits.
        ux, uy = self.make_controls_feasible_scalar(ux, uy)

        # Do simulation step, i.e. update agent with computed control input.
        px, py, vx, vy = self.dynamics_scalar(px, py, vx, vy, ux, uy, dt)
        return (px, py, vx, vy), (ux, uy)

    def control_limits(self) -> Tuple[float, float]:
        """
        .. math:: [- a_{max}, a_{max}]
        """
        return -self.acceleration_max, self.acceleration_max

    ###########################################################################
    # Agent Properties ########################################################
    ###########################################################################
    @staticmethod
    def agent_type() -> str:
        return "double_int"
