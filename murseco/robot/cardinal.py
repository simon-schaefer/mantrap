from typing import Any, Dict, List

from murseco.robot.abstract import DiscreteTimeRobot

import numpy as np

from murseco.utility.misc import cardinal_directions
from murseco.utility.stats import Distribution2D


class CardinalDiscreteTimeRobot(DiscreteTimeRobot):
    """The CardinalRobot is constrained to cardinal movements only, in order to keep the action space very small.

    :argument position: initial two-dimensional position of the robot.
    :argument thorizon: number of discrete time-steps to plan.
    :argument velocity: step width in dynamics.
    :argument policy: series of actions for the planning horizon (optional).
    """

    def __init__(
        self,
        position: np.ndarray = np.zeros(2),
        thorizon: int = 10,
        velocity: float = 0.5,
        policy: np.ndarray = None,
        **kwargs
    ):
        kwargs.update({"name": "robot/cardinal/CardinalRobot"})
        super(CardinalDiscreteTimeRobot, self).__init__(position, thorizon, policy, **kwargs)
        assert velocity > 0, "step-width must be larger than 0"

        self._velocity = velocity

    def update_policy(self, pdf: Distribution2D):
        super(CardinalDiscreteTimeRobot, self).update_policy(pdf)
        return np.zeros((self.planning_horizon, 1))

    def dynamics(self, action: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        assert action.size == 1, "action is one-dimensional for cardinal robot"
        state = super(CardinalDiscreteTimeRobot, self).dynamics(action, state)
        return state + cardinal_directions()[int(action), :] * self._velocity

    def summary(self) -> Dict[str, Any]:
        summary = super(CardinalDiscreteTimeRobot, self).summary()
        summary.update({"velocity": self._velocity})
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(CardinalDiscreteTimeRobot, cls).from_summary(json_text)
        summary.update({"velocity": float(json_text["velocity"])})
        return cls(**summary)
