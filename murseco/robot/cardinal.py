from typing import Any, Dict, List

from murseco.robot.abstract import DiscreteTimeRobot

import numpy as np

from murseco.utility.misc import cardinal_directions
from murseco.utility.stats import Distribution2D


class CardinalDiscreteTimeRobot(DiscreteTimeRobot):
    """The CardinalRobot is constrained to cardinal movements only, in order to keep the action space very small.

    :argument position: initial two-dimensional position of the robot.
    :argument thorizon: number of discrete time-steps to plan.
    :argument pstep: step width in dynamics.
    :argument policy: series of actions for the planning horizon (optional).
    """

    def __init__(self, position: np.ndarray, thorizon: int, pstep: float = 0.5, policy: np.ndarray = None):
        super(CardinalDiscreteTimeRobot, self).__init__("robot/cardinal/CardinalRobot", position, thorizon, policy)
        assert pstep > 0, "step-width must be larger than 0"

        self._pstep = pstep

    def update_policy(self, tpdf: List[Distribution2D]):
        super(CardinalDiscreteTimeRobot, self).update_policy(tpdf)
        return np.zeros((self.planning_horizon, 1))

    def dynamics(self,  action: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        assert action.size == 1, "action is one-dimensional for cardinal robot"
        state = super(CardinalDiscreteTimeRobot, self).dynamics(action, state)
        return state + cardinal_directions()[int(action), :] * self._pstep

    def summary(self) -> Dict[str, Any]:
        summary = super(CardinalDiscreteTimeRobot, self).summary()
        summary.update({"pstep": self._pstep})
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        super(CardinalDiscreteTimeRobot, cls).from_summary(json_text)
        position = np.reshape(np.array(json_text["state"]), (2,))
        thorizon = int(json_text["thorizon"])
        policy = np.reshape(np.array(json_text["policy"]), (thorizon, 1))
        return cls(position, thorizon, float(json_text["pstep"]), policy)
