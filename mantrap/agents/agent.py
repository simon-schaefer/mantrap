from abc import abstractmethod
import logging
import random
import string
from typing import Tuple

import numpy as np
import torch

from mantrap.constants import agent_speed_max
from mantrap.utility.shaping import check_ego_path, check_ego_controls, check_ego_trajectory, check_state
from mantrap.utility.utility import expand_state_vector


class Agent:
    def __init__(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        time: float = 0,
        history: torch.Tensor = None,
        identifier: str = None,
        **kwargs,
    ):
        assert position.size() == torch.Size([2]), "position must be two-dimensional (x, y)"
        assert velocity.size() == torch.Size([2]), "velocity must be two-dimensional (vx, vy)"
        assert time >= 0, "time must be larger or equal to zero"

        self._position = position.double()
        self._velocity = velocity.double()

        # Initialize (and/or append) history vector.
        if history is not None:
            assert history.shape[1] == 5, "history should contain 2D stamped position & velocity (x, y, vx, vy, t)"
            history = history.double()
            self._history = torch.stack((history, expand_state_vector(self.state, time).unsqueeze(0)), dim=1)
        else:
            self._history = expand_state_vector(self.state, time=time).view(1, 5).double()

        # Create random agent color (reddish), for evaluation only.
        self._color = np.random.uniform(0.0, 0.8, size=3).tolist()
        # Random identifier.
        letters = string.ascii_lowercase
        self._id = identifier if identifier is not None else "".join(random.choice(letters) for _ in range(3))
        logging.debug(f"agent [{self._id}]: position={self.position}, velocity={self.velocity}, color={self._color}")

    def update(self, action: torch.Tensor, dt: float):
        """Update internal state (position, velocity and history) by executing some action for time dt."""
        assert dt > 0.0, "time-step must be larger than 0"
        state_new = self.dynamics(self.state, action, dt=dt)
        self._position = state_new[0:2]
        self._velocity = state_new[2:4]

        # maximal speed constraint.
        if self.speed > agent_speed_max:
            logging.error(f"agent {self.id} has surpassed maximal speed, with {self.speed} > {agent_speed_max}")
            assert not torch.isinf(self.speed), "speed is infinite, physical break"
            self._velocity = self._velocity / self.speed * agent_speed_max

        # append history with new state.
        state_new = expand_state_vector(self.state, time=self._history[-1, -1].item() + dt).unsqueeze(0)
        self._history = torch.cat((self._history, state_new), dim=0)

    def unroll_trajectory(self, controls: torch.Tensor, dt: float) -> torch.Tensor:
        """Build the trajectory from some controls and current state, by iteratively applying the model dynamics.
        Thereby a perfect model i.e. without uncertainty and correct is assumed.

        :param controls: sequence of inputs to apply to the robot (N, input_size).
        :param dt: time interval [s].
        :return: resulting trajectory (no uncertainty in dynamics assumption !), (N, 4).
        """
        assert dt > 0.0, "time-step must be larger than 0"
        if len(controls.shape) == 1:  # singe action as policy
            controls = controls.unsqueeze(dim=0)

        # initial trajectory point is the current state.
        trajectory = torch.zeros((controls.shape[0] + 1, 5))
        trajectory[0, :] = self.state_with_time

        # every next state follows from robot's dynamics recursion, basically assuming no model uncertainty.
        for k in range(controls.shape[0]):
            state_k = self.dynamics(trajectory[k, :], action=controls[k, :], dt=dt)
            trajectory[k + 1, :] = torch.cat((state_k, torch.ones(1) * trajectory[k, -1] + dt))

        assert check_ego_trajectory(trajectory, t_horizon=controls.shape[0] + 1, pos_and_vel_only=False)
        return trajectory

    def roll_trajectory(self, trajectory: torch.Tensor, dt: float) -> torch.Tensor:
        """Determine the controls by iteratively applying the agent's model inverse dynamics.
        Thereby a perfect model i.e. without uncertainty and correct is assumed.

        :param trajectory: sequence of states to apply to the robot (N, 4).
        :param dt: time interval [s].
        :return: inferred controls (no uncertainty in dynamics assumption !), (N, input_size).
        """
        assert dt > 0.0, "time-step must be larger than 0"
        assert check_ego_trajectory(trajectory, pos_and_vel_only=True)

        # every control input follows from robot's dynamics recursion, basically assuming no model uncertainty.
        controls = torch.zeros((trajectory.shape[0] - 1, 2))  # assuming input_size = 2
        for k in range(trajectory.shape[0] - 1):
            controls[k, :] = self.inverse_dynamics(trajectory[k + 1, :], trajectory[k, :], dt=dt)

        assert check_ego_controls(controls, t_horizon=trajectory.shape[0] - 1)
        return controls

    @staticmethod
    def expand_trajectory(path: torch.Tensor, dt: float, t_start: float = 0.0) -> torch.Tensor:
        """Derive (position, orientation, velocity)-trajectory information from position data only, using naive
        discrete differentiation, i.e. v_i = (x_i+1 - x_i) / dt. """
        assert check_ego_path(path)

        t_horizon = path.shape[0]
        trajectory = torch.zeros((t_horizon, 5))

        trajectory[:, 0:2] = path
        trajectory[:-1, 2:4] = (trajectory[1:, 0:2] - trajectory[0:-1, 0:2]) / dt
        trajectory[:, 4] = torch.linspace(t_start, t_start + t_horizon * dt, steps=t_horizon)

        assert check_ego_trajectory(trajectory, t_horizon=t_horizon, pos_and_vel_only=True)
        return trajectory

    def reset(self, state: torch.Tensor, history: torch.Tensor = None):
        """Reset the complete state of the agent by resetting its position and velocity. Either adapt the agent's
        history to the new state (i.e. append it to the already existing history) if history is given as None or set
        it to some given trajectory.
        """
        assert check_state(state, enforce_temporal=True), "state has to be at least 5-dimensional"
        if history is None:
            history = self._history = torch.cat((self._history, state.unsqueeze(0)), dim=0)
        self._position = state[0:2]
        self._velocity = state[2:4]
        self._history = history

    @abstractmethod
    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float) -> torch.Tensor:
        """Forward integrate the egos motion given some state-action pair and an integration time-step. Since every
        agent type has different dynamics (like single-integrator or Dubins Car) this method is implemented
        abstractly.
        """
        raise NotImplementedError

    @abstractmethod
    def inverse_dynamics(self, state: torch.Tensor, state_previous: torch.Tensor, dt: float) -> torch.Tensor:
        """Determine the ego motion given its current and previous state. Since every agent type has different dynamics
        (like single-integrator or Dubins Car) this method is implemented abstractly.
        """
        raise NotImplementedError

    @abstractmethod
    def control_limits(self) -> Tuple[float, float]:
        raise NotImplementedError

    ###########################################################################
    # Computation graph #######################################################
    ###########################################################################
    def detach(self):
        self._position = self._position.detach()
        self._velocity = self._velocity.detach()

    ###########################################################################
    # State properties ########################################################
    ###########################################################################
    @property
    def state(self) -> torch.Tensor:
        return torch.cat((self.position, self.velocity))

    @property
    def state_with_time(self) -> torch.Tensor:
        return torch.cat((self.position, self.velocity, torch.ones(1) * self._history[-1, -1]))

    @property
    def position(self) -> torch.Tensor:
        return self._position

    @property
    def velocity(self) -> torch.Tensor:
        return self._velocity

    @property
    def speed(self) -> float:
        return torch.norm(self._velocity)

    ###########################################################################
    # History properties ######################################################
    ###########################################################################
    @property
    def history(self) -> torch.Tensor:
        return self._history

    ###########################################################################
    # Visualization/Utility properties ########################################
    ###########################################################################
    @property
    def color(self) -> np.ndarray:
        return self._color

    @property
    def id(self) -> str:
        return self._id
