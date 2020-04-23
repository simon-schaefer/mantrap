from abc import ABC, abstractmethod
import logging
import random
import string
from typing import Tuple

import numpy as np
import torch

from mantrap.constants import *
from mantrap.utility.maths import Shape2D
from mantrap.utility.shaping import (
    check_ego_path,
    check_ego_action,
    check_ego_controls,
    check_ego_trajectory,
    check_ego_state
)


class Agent(ABC):
    """ General agent representation.

    An agent, whether in environment or real world, has a five-dimensional state vector and a state history, which are
    defined as follows:

    .. math:: s_t = (pos_x(t), pos_y(t), vel_x(t), vel_y(t), time)
    .. math:: history = (s_{t0}, s_{t1}, ..., s_{t})

    For unique identification each agent has an id (3-character string) and for visualisation purposes a unique color.
    Both are created randomly during initialization.
    The internal state of the agent can be altered by calling the `update()` function, which uses some (control) input
    to update the agent's state using its dynamics, or the `reset()` function, which sets the internal state to some
    value directly. All other methods do not alter the internal state.

    :param position: current 2D position vector (2).
    :param velocity: current 2D velocity vector (2).
    :param time: current time stamp, default = 0.0.
    :param history: current agent's state history (N, 5), default = no history.
    :param identifier: agent's pre-set identifier, default = none so initialized randomly during initialization.
    """

    def __init__(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor = torch.zeros(2),
        time: float = 0,
        history: torch.Tensor = None,
        is_robot: bool = False,
        identifier: str = None,
        **unused
    ):
        assert position.size() == torch.Size([2]), "position must be two-dimensional (x, y)"
        assert velocity.size() == torch.Size([2]), "velocity must be two-dimensional (vx, vy)"
        assert time >= 0, "time must be larger or equal to zero"

        self._state = torch.cat((position.float(), velocity.float(), torch.ones(1) * time))

        # Initialize (and/or append) history vector. The current state must be at the end of the internal history,
        # so either append it or create it when not already the case.
        state_un_squeezed = self.state_with_time.unsqueeze(dim=0)
        if history is not None:
            assert history.shape[1] == 5
            history = history.float()

            if not torch.all(torch.isclose(history[-1, :], state_un_squeezed)):
                self._history = torch.cat((history, state_un_squeezed), dim=0)
            else:
                self._history = history
        else:
            self._history = state_un_squeezed

        # Initialize agent properties.
        self._is_robot = is_robot
        self._max_speed = AGENT_SPEED_MAX if not is_robot else ROBOT_SPEED_MAX
        self._max_acceleration = AGENT_ACC_MAX if not is_robot else ROBOT_ACC_MAX

        # Create random agent color (reddish), for evaluation only.
        self._color = np.random.uniform(0.0, 0.8, size=3).tolist()
        # Random identifier.
        letters = string.ascii_lowercase
        self._id = identifier if identifier is not None else "".join(random.choice(letters) for _ in range(3))
        logging.debug(f"agent [{self._id}]: position={self.position}, velocity={self.velocity}, color={self._color}")

    def update(self, action: torch.Tensor, dt: float):
        """Update internal state (position, velocity and history) by forward integrating the agent's dynamics over
        the given time-step. The new state and time are then appended to the agent's state history.

        :param action: control input (size depending on agent type).
        :param dt: forward integration time step [s].
        """
        assert check_ego_action(x=action)
        assert dt > 0.0

        # Maximal control effort constraint.
        _, control_limit = self.control_limits()
        control_effort = torch.norm(action)
        if control_effort > control_limit:
            logging.warning(f"agent {self.id} has surpassed control limit, with {control_effort} > {control_limit}")
            assert not torch.isinf(control_effort), "control effort is infinite, physical break"
            action = action / control_effort * control_limit

        # Compute next state using internal dynamics.
        state_new = self.dynamics(self.state_with_time, action, dt=dt)

        # Update internal state and append history with new state.
        self._state = state_new
        self._history = torch.cat((self._history, state_new.unsqueeze(0)), dim=0)

        # Perform sanity check for agent properties.
        assert self.sanity_check()

    def reset(self, state: torch.Tensor, history: torch.Tensor = None):
        """Reset the complete state of the agent by resetting its position and velocity. Either adapt the agent's
        history to the new state (i.e. append it to the already existing history) if history is given as None or set
        it to some given trajectory.

        :param state: new state (5).
        :param history: new state history (N, 5).
        """
        assert check_ego_state(state, enforce_temporal=True), "state has to be at least 5-dimensional"

        self._state = state
        self._history = torch.cat((self._history, state.unsqueeze(0)), dim=0) if history is None else history

        # Perform sanity check for agent properties.
        assert self.sanity_check()

    ###########################################################################
    # Trajectory ##############################################################
    ###########################################################################
    def unroll_trajectory(self, controls: torch.Tensor, dt: float) -> torch.Tensor:
        """Build the trajectory from some controls and current state, by iteratively applying the model dynamics.
        Thereby a perfect model i.e. without uncertainty and correct is assumed.

        :param controls: sequence of inputs to apply to the robot (N, input_size).
        :param dt: time interval [s].
        :return: resulting trajectory (no uncertainty in dynamics assumption !), (N, 4).
        """
        assert check_ego_controls(x=controls)
        assert dt > 0.0

        if len(controls.shape) == 1:  # singe action as policy
            controls = controls.unsqueeze(dim=0)

        # initial trajectory point is the current state.
        trajectory = torch.zeros((controls.shape[0] + 1, 5))
        trajectory[0, :] = self.state_with_time

        # every next state follows from robot's dynamics recursion, basically assuming no model uncertainty.
        for k in range(controls.shape[0]):
            trajectory[k + 1, :] = self.dynamics(trajectory[k, :], action=controls[k, :], dt=dt)

        assert check_ego_trajectory(trajectory, t_horizon=controls.shape[0] + 1, pos_and_vel_only=False)
        return trajectory

    def roll_trajectory(self, trajectory: torch.Tensor, dt: float) -> torch.Tensor:
        """Determine the controls by iteratively applying the agent's model inverse dynamics.
        Thereby a perfect model i.e. without uncertainty and correct is assumed.

        :param trajectory: sequence of states to apply to the robot (N, 4).
        :param dt: time interval [s].
        :return: inferred controls (no uncertainty in dynamics assumption !), (N, input_size).
        """
        assert check_ego_trajectory(trajectory, pos_and_vel_only=True)
        assert dt > 0.0

        # every control input follows from robot's dynamics recursion, basically assuming no model uncertainty.
        controls = torch.zeros((trajectory.shape[0] - 1, 2))  # assuming input_size = 2
        for k in range(trajectory.shape[0] - 1):
            controls[k, :] = self.inverse_dynamics(trajectory[k + 1, :], trajectory[k, :], dt=dt)

        assert check_ego_controls(controls, t_horizon=trajectory.shape[0] - 1)
        return controls

    def expand_trajectory(self, path: torch.Tensor, dt: float) -> torch.Tensor:
        """Derive (position, orientation, velocity)-trajectory information from position data only, using naive
        discrete differentiation, i.e. v_i = (x_i+1 - x_i) / dt.

        :param path: sequence of states (position, velocity) without temporal dimension (N, 4).
        :param dt: time interval which is assumed to be constant over full path sequence [s].
        :returns trajectory: temporally-expanded path (N, 5).s
        """
        assert check_ego_path(path)
        assert dt > 0.0

        t_horizon = path.shape[0]
        t_start = float(self.state_with_time[-1])
        trajectory = torch.zeros((t_horizon, 5))

        trajectory[:, 0:2] = path
        trajectory[:-1, 2:4] = (trajectory[1:, 0:2] - trajectory[0:-1, 0:2]) / dt
        trajectory[:, 4] = torch.linspace(t_start, t_start + (t_horizon - 1) * dt, steps=t_horizon)

        assert check_ego_trajectory(trajectory, t_horizon=t_horizon)
        return trajectory

    ###########################################################################
    # Reachability ############################################################
    ###########################################################################
    @abstractmethod
    def reachability_boundary(self, time_steps: int, dt: float) -> Shape2D:
        """Compute the forward reachable set within the time-horizon with `time_steps` number of discrete
        steps with time-step `dt`. While the reachable set can be computed numerically in general, an
        exact (and more efficient) implementation has to be done in the child classes. Return this exact
        solution as class object of `Shape2D` type.

        :param time_steps: number of discrete time-steps in reachable time-horizon.
        :param dt: time interval which is assumed to be constant over full path sequence [s].
        """
        raise NotImplementedError

    ###########################################################################
    # Dynamics ################################################################
    ###########################################################################
    @abstractmethod
    def dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float) -> torch.Tensor:
        """Forward integrate the egos motion given some state-action pair and an integration time-step. Passing the
        state, instead of using the internal state, allows the method to be used for other state vector than the
        internal state, e.g. for forward predicting over a time horizon > 1.
        Since every agent type has different dynamics (like single-integrator or Dubins Car) this method is
        implemented abstractly.

        :param state: state to be updated @ t = k (5).
        :param action: control input @ t = k (size depending on agent type).
        :param dt: forward integration time step [s].
        :returns: updated state vector with time @ t = k + dt (5).
        """
        raise NotImplementedError

    @abstractmethod
    def inverse_dynamics(self, state: torch.Tensor, state_previous: torch.Tensor, dt: float) -> torch.Tensor:
        """Determine the ego motion given its current and previous state. Passing the state, instead of using the
        internal state, allows the method to be used for other state vector than the internal state, e.g. for forward
        predicting over a time horizon > 1.
        Since every agent type has different dynamics (like single-integrator or Dubins Car) this method is
        implemented abstractly.

        :param state: state @ t = k (4 or 5).
        :param state_previous: previous state @ t = k - dt (4 or 5).
        :param dt: forward integration time step [s].
        :returns: control input @ t = k (size depending on agent type).
        """
        raise NotImplementedError

    @abstractmethod
    def control_limits(self) -> Tuple[float, float]:
        """Returns agent's control input limitations, i.e. lower and upper bound."""
        raise NotImplementedError

    ###########################################################################
    # Computation graph #######################################################
    ###########################################################################
    def detach(self):
        """Detach the agent's internal variables (position, velocity, history) from computation tree. This is sometimes
        required to completely separate subsequent computations in PyTorch."""
        self._state = self._state.detach()
        self._history = self._history.detach()

    ###########################################################################
    # Utility #################################################################
    ###########################################################################
    @staticmethod
    def expand_state_vector(state_4: torch.Tensor, time: float) -> torch.Tensor:
        """Expand 4 dimensional (x, y, vx, vy) state vector by time information. """
        assert state_4.size() == torch.Size([4])

        state = torch.cat((state_4, torch.ones(1) * time))
        assert check_ego_state(state, enforce_temporal=True)
        return state

    ###########################################################################
    # Operators ###############################################################
    ###########################################################################
    def __eq__(self, other, check_class: bool = True):
        """Two agents are considered as being identical when their current state and their complete state history are
        equal. However the agent's descriptive parameters such as its id are not part of this comparison, since they
        are initialized as random and not descriptive for the state an agent is in. Also the agent's class does not
        have to be the same (given flag).
        """
        assert self.id == other.id
        if check_class:
            assert self.__class__ == other.__class__
        assert torch.all(torch.isclose(self.state_with_time, other.state_with_time))
        assert torch.all(torch.isclose(self.history, other.history))
        return True

    def sanity_check(self) -> bool:
        """Sanity check for agent.
        In order to evaluate the sanity of the agent in the most general form, several internal states such as the
        position and velocity are checked to be of the right type. Also the history should always reflect the
        states until and including (!) the current state. """
        assert self.position is not None
        assert self.velocity is not None
        assert self.history is not None

        assert check_ego_state(x=self.state_with_time, enforce_temporal=True)
        assert check_ego_trajectory(x=self.history)
        assert torch.all(torch.isclose(self.history[-1, :], self.state_with_time))
        return True

    ###########################################################################
    # State properties ########################################################
    ###########################################################################
    @property
    def state(self) -> torch.Tensor:
        return self._state[0:4]

    @property
    def state_with_time(self) -> torch.Tensor:
        return self._state

    @property
    def position(self) -> torch.Tensor:
        return self._state[0:2]

    @property
    def velocity(self) -> torch.Tensor:
        return self._state[2:4]

    @property
    def speed(self) -> float:
        return torch.norm(self.velocity)

    ###########################################################################
    # History properties ######################################################
    ###########################################################################
    @property
    def history(self) -> torch.Tensor:
        return self._history

    ###########################################################################
    # Agent properties ########################################################
    ###########################################################################
    @property
    def speed_max(self) -> float:
        return self._max_speed

    @property
    def acceleration_max(self) -> float:
        return self._max_acceleration

    @property
    def is_robot(self) -> bool:
        return self._is_robot

    ###########################################################################
    # Visualization/Utility properties ########################################
    ###########################################################################
    @property
    def color(self) -> np.ndarray:
        return self._color

    @property
    def id(self) -> str:
        return self._id
