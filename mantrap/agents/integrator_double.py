import numpy as np

import mantrap.constants
from mantrap.agents.agent import Agent


class DoubleIntegratorDTAgent(Agent):
    def __init__(self, position: np.ndarray, velocity: np.ndarray = np.zeros(2), history: np.ndarray = None, **kwargs):
        super(DoubleIntegratorDTAgent, self).__init__(position, velocity, history=history, **kwargs)

    def dynamics(self, state: np.ndarray, action: np.ndarray, dt: float = mantrap.constants.sim_dt_default):
        assert state.size == 5, "state should be two-dimensional (x, y, theta, vx, vy)"
        assert action.size == 2, "action must be two-dimensional (vx, vy)"

        velocity_new = state[3:5] + action * dt
        theta_new = np.arctan2(velocity_new[1], velocity_new[0])
        position_new = state[0:2] + state[3:5] * dt + 0.5 * action * dt ** 2
        return np.hstack((position_new, theta_new, velocity_new))
