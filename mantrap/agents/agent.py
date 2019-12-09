from abc import abstractmethod
import logging
import random
import string

import numpy as np

from mantrap.constants import sim_speed_max, sim_dt_default


class Agent:
    def __init__(self, position: np.ndarray, velocity: np.ndarray = np.zeros(2), history: np.ndarray = None, **kwargs):
        assert position.size == 2, "position must be two-dimensional (x, y)"
        assert velocity.size == 2, "velocity must be two-dimensional (vx, vy)"

        self._position = position
        self._velocity = velocity

        # Initialize (and/or append) history vector.
        if history is not None:
            assert history.shape[1] == 6, "history should contain 2D stamped pose & velocity (x, y, theta, vx, vy, t)"
            self._history = history
            self._history = np.vstack((self._history, np.hstack((self.state, 0))))
        else:
            self._history = np.reshape(np.hstack((self.state, 0)), (1, 6))

        # Create random agent color (reddish), for evaluation only.
        self._color = np.hstack((1.0, np.random.uniform(0.0, 0.5, size=2)))
        # Random identifier.
        letters = string.ascii_lowercase
        self._id = "".join(random.choice(letters) for i in range(3))
        logging.debug(f"agent: position={self.position}, velocity={self.velocity}, id={self._id}, color={self._color}")

    def update(self, action: np.ndarray, dt: float = sim_dt_default):
        """Update internal state (position, velocity and history) by executing some action for time dt."""
        assert dt > 0.0, "time-step must be larger than 0"
        state_new = self.dynamics(self.state, action, dt=dt)
        self._position = state_new[0:2]
        self._velocity = state_new[3:5]

        # maximal speed constraint.
        if self.speed > sim_speed_max:
            logging.warning(f"agent {self.id} has surpassed maximal speed, with {self.speed} > {sim_speed_max}")
            assert not np.isinf(self.speed), "speed is infinite, physical break"
            self._velocity = self._velocity / self.speed * sim_speed_max

        # append history with new state.
        self._history = np.vstack((self._history, np.hstack((self.state, self._history[-1, -1] + dt))))

    def unroll_trajectory(self, policy: np.ndarray, dt: float = sim_dt_default) -> np.ndarray:
        """Build the trajectory from some policy and current state, by iteratively applying the model dynamics.
        Thereby a perfect model i.e. without uncertainty and correct is assumed.

        :param policy: sequence of inputs to apply to the robot (N, input_size).
        :param dt: time interval [s].
        :return: resulting trajectory (no uncertainty in dynamics assumption !), (N, 4).
        """
        assert dt > 0.0, "time-step must be larger than 0"
        if len(policy.shape) == 1:  # singe action as policy
            policy = np.expand_dims(policy, axis=0)

        # initial trajectory point is the current state.
        trajectory = np.zeros((policy.shape[0] + 1, 6))
        trajectory[0, :] = np.hstack((self.state, 0))

        # every next state follows from robot's dynamics recursion, basically assuming no model uncertainty.
        state_at_t = self.state.copy()
        for k in range(policy.shape[0]):
            state_at_t = self.dynamics(state_at_t, action=policy[k, :], dt=dt)
            trajectory[k + 1, :] = np.hstack((state_at_t, (k + 1) * dt))
        return trajectory

    def reset(self, position: np.ndarray, velocity: np.ndarray, history: np.ndarray = None):
        """Reset the complete state of the agent by resetting its position and velocity. Additionally, the history can
        be resetted, if given as None then it is re-initialized. As the agent is fully resetted it gets a new id and
        color assigned to it (!).
        """
        self.__init__(position=position, velocity=velocity, history=history)

    @abstractmethod
    def dynamics(self, state: np.ndarray, action: np.ndarray, dt: float = sim_dt_default):
        """Forward integrate the egos motion given some state-action pair and an integration time-step. Since every
        ego agent type has different dynamics (like single-integrator or Dubins Car) this method is implemented
        abstractly.
        """
        pass

    ###########################################################################
    # State properties ########################################################
    ###########################################################################

    @property
    def state(self) -> np.ndarray:
        return np.hstack((self.pose, self.velocity))

    @property
    def position(self) -> np.ndarray:
        return self._position

    @property
    def pose(self) -> np.ndarray:
        return np.array([self._position[0], self._position[1], np.arctan2(self._velocity[1], self._velocity[0])])

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity

    @property
    def speed(self) -> float:
        return np.linalg.norm(self._velocity)

    ###########################################################################
    # Dimensions ##############################################################
    ###########################################################################

    @property
    def length_state(self) -> int:
        return 5  # (x, y, theta, vx, vy)

    @property
    def length_action(self) -> int:
        return 2  # override if necessary (!)

    ###########################################################################
    # History properties ######################################################
    ###########################################################################

    @property
    def history(self) -> np.ndarray:
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
