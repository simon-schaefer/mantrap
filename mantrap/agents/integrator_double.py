from typing import Tuple

import torch

from mantrap.agents.agent import Agent
from mantrap.utility.maths import Circle
from mantrap.utility.shaping import check_ego_action, check_ego_state


class DoubleIntegratorDTAgent(Agent):

    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float) -> torch.Tensor:
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
