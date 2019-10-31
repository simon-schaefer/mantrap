from abc import abstractmethod, abstractproperty
from typing import Any, Dict, Union

import numpy as np

from murseco.utility.io import JSONSerializer


class DTRobot(JSONSerializer):
    def __init__(self, state: np.ndarray, **kwargs):
        super(DTRobot, self).__init__(**kwargs)
        assert state.size >= 2, "state vector = (x, y, ...) with (x, y) being the initial position"

        self._state = state

    def trajectory(self, policy: np.ndarray) -> Union[np.ndarray, None]:
        """Build the trajectory from the internal policy and current state, by iteratively applying the model dynamics.
        Thereby a perfect model i.e. without uncertainty and correct is assumed. If no policy has been determined yet,
        return None instead of the trajectory.

        :return trajectory: array of ordered positions of the robot over the planning horizon (thorizon, 2).
        """
        trajectory = np.zeros((policy.shape[0] + 1, 2))

        # initial trajectory point is the current state.
        trajectory[0, :] = self.position

        # every next state follows from robot's dynamics recursion, basically assuming no model uncertainty.
        state = self.state.copy()
        for i in range(policy.shape[0]):
            state = self.dynamics(policy[i, :], state)
            trajectory[i + 1, :] = state[:2].copy()
        return trajectory

    @property
    def position(self) -> np.ndarray:
        return self._state[:2]

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def state_size(self) -> int:
        return self._state.size

    @property
    @abstractmethod
    def input_size(self) -> int:
        return -1

    @abstractmethod
    def dynamics(self, action: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        return self.state if state is None else state

    @abstractmethod
    def summary(self) -> Dict[str, Any]:
        summary = super(DTRobot, self).summary()
        summary.update({"state": self._state.tolist()})
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(DTRobot, cls).from_summary(json_text)
        position = np.reshape(np.array(json_text["state"]), (2,))
        summary.update({"position": position})
        return summary
