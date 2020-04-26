import math
from typing import Tuple

import torch

from mantrap.agents.agent import Agent
from mantrap.utility.maths import Circle
from mantrap.utility.shaping import check_ego_action, check_ego_state


class IntegratorDTAgent(Agent):

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt:  float) -> torch.Tensor:
        """
        .. math:: vel_{t+1} = action
        .. math:: pos_{t+1} = pos_t + vel_{t+1} * dt
        """
        assert check_ego_state(state, enforce_temporal=True)  # (x, y, theta, vx, vy, t)
        assert action.size() == torch.Size([2])  # (vx, vy)
        action = action.float()

        state_new = torch.zeros(5)
        state_new[2:4] = action  # velocity
        state_new[0:2] = state[0:2] + state_new[2:4] * dt
        state_new[4] = state[-1] + dt

        assert check_ego_state(state_new, enforce_temporal=True)
        return state_new

    @staticmethod
    def dynamics_scalar(px: float, py: float, vx: float, vy: float, ux: float, uy: float, dt: float
                        ) -> Tuple[float, float, float, float]:
        return px + ux * dt, py + uy * dt, ux, uy

    def inverse_dynamics(self, state: torch.Tensor, state_previous: torch.Tensor, dt: float) -> torch.Tensor:
        """
        .. math:: action = (pos_t - pos_{t-1}) / dt
        """
        assert check_ego_state(state, enforce_temporal=False)
        assert check_ego_state(state_previous, enforce_temporal=False)

        action = (state[0:2] - state_previous[0:2]) / dt
        assert check_ego_action(x=action)
        return action

    def go_to_point(
        self,
        state: Tuple[float, float, float, float],
        target_point: Tuple[float, float],
        speed: float,
        dt: float,
    ) -> Tuple[Tuple[float, float, float, float], Tuple[float, float]]:
        """Determine and execute the controls for going for the given state to some target point with respect to
        the internal dynamics.

        Due to the instant dynamics of a single integrator going from one point to another simple means instantly
        adapting the controls to go in the right direction.

        :param state: pos_and_vel_only state vector i.e. (px, py, vx, vy) for starting state.
        :param target_point: 2D target point (px, py).
        :param speed: preferable speed for state update [m/s].
        :param dt: update time interval [s].
        :return updated state at t = t0 + dt and used (cardinal) control input.
        """
        px, py, vx, vy = state
        target_point_x, target_point_y = target_point

        # Determine cartesian controls with "infinite" speed. Then adapt the controls to the given reference
        # speed value (while keeping the direction).
        ux, uy = target_point_x - px, target_point_y - py
        u_speed = math.hypot(ux, uy)
        ux, uy = ux / u_speed * speed, uy / u_speed * speed

        # Do simulation step, i.e. update agent with computed control input.
        px, py, vx, vy = self.dynamics_scalar(px, py, vx, vy, ux, uy, dt)
        return (px, py, vx, vy), (ux, uy)

    def control_limits(self) -> Tuple[float, float]:
        """
        .. math:: [- v_{max}, v_{max}]
        """
        return -self.speed_max, self.speed_max

    ###########################################################################
    # Reachability ############################################################
    ###########################################################################
    def reachability_boundary(self, time_steps: int, dt: float) -> Circle:
        """Single integrators can adapt their velocity instantly in any direction. Therefore the forward
        reachable set within the number of time_steps is just a circle (in general ellipse, but agent is
        assumed to be isotropic within this class, i.e. same control bounds for both x- and y-direction)
        around the current position, with radius being the maximal allowed agent speed.
        With `T = time_steps * dt` being the time horizon, the reachability bounds are determined for,
        the circle has the following parameters:

        .. math:: center = x(0)
        .. math:: radius = v_{max} * T

        :param time_steps: number of discrete time-steps in reachable time-horizon.
        :param dt: time interval which is assumed to be constant over full path sequence [s].
        """
        return Circle(center=self.position, radius=self.speed_max * dt * time_steps)

    ###########################################################################
    # Agent Properties ########################################################
    ###########################################################################
    @property
    def acceleration_max(self) -> float:
        """Since single integrators are assumed to react instantly, their maximal acceleration
        is in fact infinite !"""
        return math.inf
