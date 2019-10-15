from abc import abstractmethod
import logging
from typing import Any, Dict, Union

import numpy as np

from murseco.utility.io import JSONSerializer
from murseco.utility.stats import Distribution2D


class DTRobot(JSONSerializer):
    def __init__(self, state: np.ndarray, thorizon: int, policy: np.ndarray = None, **kwargs):
        super(DTRobot, self).__init__(**kwargs)
        assert state.size >= 2, "state vector = (x, y, ...) with (x, y) being the initial position"
        if policy is not None:
            assert policy.shape[0] == thorizon, "policy must contain steps equal to planning horizon (thorizon)"

        self._state = state
        self._policy = policy
        self._thorizon = thorizon

    @property
    def position(self) -> np.ndarray:
        return self._state[:2]

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def trajectory(self) -> Union[np.ndarray, None]:
        """Build the trajectory from the internal policy and current state, by iteratively applying the model dynamics.
        Thereby a perfect model i.e. without uncertainty and correct is assumed. If no policy has been determined yet,
        return None instead of the trajectory.

        :return trajectory: array of ordered positions of the robot over the planning horizon (thorizon, 2).
        """
        if self._policy is None:
            return np.tile(self.position, (self.planning_horizon, 2))
        trajectory = np.zeros((self._policy.shape[0] + 1, 2))

        # initial trajectory point is the current state.
        trajectory[0, :] = self.position

        # every next state follows from robot's dynamics recursion, basically assuming no model uncertainty.
        state = self.state.copy()
        for i in range(self._policy.shape[0]):
            state = self.dynamics(self._policy[i, :], state)
            trajectory[i + 1, :] = state[:2].copy()
        return trajectory

    @property
    def policy(self) -> np.ndarray:
        return self._policy

    @property
    def planning_horizon(self) -> int:
        return self._thorizon

    @abstractmethod
    def dynamics(self, action: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        return self.state if state is None else state

    @abstractmethod
    def update_policy(self, pdf: Distribution2D):
        logging.debug("update_policy -> starting")
        pass

    @abstractmethod
    def summary(self) -> Dict[str, Any]:
        summary = super(DTRobot, self).summary()
        policy = self._policy.tolist() if self._policy is not None else None
        summary.update({"state": self._state.tolist(), "thorizon": self._thorizon, "policy": policy})
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(DTRobot, cls).from_summary(json_text)
        position = np.reshape(np.array(json_text["state"]), (2,))
        thorizon = int(json_text["thorizon"])
        policy = np.reshape(np.array(json_text["policy"]), (thorizon, 1)) if json_text["policy"] != "null" else None
        summary.update({"position": position, "thorizon": thorizon, "policy": policy})
        return summary
