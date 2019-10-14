from typing import Any, Dict

from murseco.robot.abstract import DTRobot

import numpy as np

from murseco.utility.stats import Distribution2D


class IntegratorDTRobot(DTRobot):
    """The IntegratorRobot can move in any direction with the action stated in the policy, unconstrained in its
    movement per default. The dynamics therefore a single-integrator dynamics.

    :argument position: initial two-dimensional position of the robot.
    :argument thorizon: number of discrete time-steps to plan.
    :argument policy: series of actions for the planning horizon (optional).
    """

    def __init__(
        self,
        position: np.ndarray = np.zeros(2),
        thorizon: int = 10,
        policy: np.ndarray = None,
        **kwargs
    ):
        kwargs.update({"name": "robot/integrator/IntegratorDTRobot"})
        super(IntegratorDTRobot, self).__init__(position, thorizon, policy, **kwargs)

    def update_policy(self, pdf: Distribution2D):
        super(IntegratorDTRobot, self).update_policy(pdf)
        return np.zeros((self.planning_horizon, 2))

    def dynamics(self, action: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        assert action.size == 2, "action is two-dimensional position change for integrator robot"
        state = super(IntegratorDTRobot, self).dynamics(action, state)
        return state + action

    def summary(self) -> Dict[str, Any]:
        summary = super(IntegratorDTRobot, self).summary()
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(IntegratorDTRobot, cls).from_summary(json_text)
        return cls(**summary)
