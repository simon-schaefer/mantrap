import numpy as np

import mantrap.constants
from mantrap.agents.agent import Agent


class IntegratorDTAgent(Agent):
    def __init__(self, position: np.ndarray, velocity: np.ndarray = np.zeros(2), history: np.ndarray = None):
        super(IntegratorDTAgent, self).__init__(position, velocity, history=history)

    def dynamics(self, state: np.ndarray, action: np.ndarray, dt: float = mantrap.constants.sim_dt_default):
        assert state.size == 5, "state should be two-dimensional (x, y, theta, vx, vy)"
        assert action.size == 2, "action must be two-dimensional (vx, vy)"

        velocity_new = action
        theta_new = np.arctan2(velocity_new[1], velocity_new[0])
        position_new = state[0:2] + velocity_new * dt
        return np.hstack((position_new, theta_new, velocity_new))
