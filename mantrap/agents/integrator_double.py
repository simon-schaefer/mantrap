import math
from typing import Tuple

import torch

from mantrap.agents.agent import Agent
from mantrap.utility.maths import Circle
from mantrap.utility.shaping import check_ego_action, check_ego_state


class DoubleIntegratorDTAgent(Agent):

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float) -> torch.Tensor:
        """
          .. math:: vel_{t+1} = vel_t + action * dt
          .. math:: pos_{t+1} = pos_t + vel_{t+1} * dt
          """
        assert check_ego_state(state, enforce_temporal=True)  # (x, y, theta, vx, vy, t)
        assert check_ego_action(action)  # (ax, ay)
        action = action.float()

        state_new = torch.zeros(5)
        state_new[2:4] = (state[2:4] + action * dt).float()  # velocity
        state_new[0:2] = (state[0:2] + state[2:4] * dt).float()  # position
        state_new[4] = state[4] + dt

        assert check_ego_state(state_new, enforce_temporal=True)
        return state_new

    @staticmethod
    def dynamics_scalar(px: float, py: float, vx: float, vy: float, ux: float, uy: float, dt: float
                        ) -> Tuple[float, float, float, float]:
        return px + vx * dt, py + vy * dt, vx + ux * dt, vy + uy * dt

    def inverse_dynamics(self, state: torch.Tensor, state_previous: torch.Tensor, dt: float) -> torch.Tensor:
        """
        .. math:: action = (vel_t - vel_{t-1}) / dt
        """
        assert check_ego_state(state, enforce_temporal=False)
        assert check_ego_state(state_previous, enforce_temporal=False)

        action = (state[2:4] - state_previous[2:4]) / dt
        assert check_ego_action(x=action)
        return action

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
        :return updated state at t = t0 + dt and used (cardinal) control input.
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
        lower, upper = self.control_limits()
        ux = min(upper, max(lower, ux))
        uy = min(upper, max(lower, uy))

        # Do simulation step, i.e. update agent with computed control input.
        px, py, vx, vy = self.dynamics_scalar(px, py, vx, vy, ux, uy, dt)
        return (px, py, vx, vy), (ux, uy)

    def control_limits(self) -> Tuple[float, float]:
        """
        .. math:: [- a_{max}, a_{max}]
        """
        return -self.acceleration_max, self.acceleration_max

    ###########################################################################
    # Reachability ############################################################
    ###########################################################################
    def reachability_boundary(self, time_steps: int, dt: float) -> Circle:
        """Double integrators cannot adapt their velocity instantly, but delayed by (instantly) changing
        their acceleration in any direction. Similarly to the single integrator therefore the forward
        reachable set within the number of time_steps is just a circle (in general ellipse, but agent is
        assumed to be isotropic within this class, i.e. same bounds for both x- and y-direction), just
        not around the current position, since the velocity the agent has acts as an "inertia", shifting
        the center of the circle. The radius of the circle results from the double integrator dynamics,
        the change of position with altering the acceleration to be exact. With `T = time_steps * dt`
        being the time horizon, the reachability bounds are determined for, the circle has the following
        parameters:

        .. math:: center = x(0) + v(0) * T
        .. math:: radius = 0.5 * a_{max} * T^2

        :param time_steps: number of discrete time-steps in reachable time-horizon.
        :param dt: time interval which is assumed to be constant over full path sequence [s].
        """
        center = self.position + self.velocity * time_steps * dt
        radius = 0.5 * self.acceleration_max * (time_steps * dt)**2
        return Circle(center=center, radius=radius)
