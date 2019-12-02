from typing import Union

import numpy as np

from mantrap.simulation.abstract import Simulation
from mantrap.solver.abstract import Solver


class SimulatedRRTSolver(Solver):
    def __init__(self, sim: Simulation, goal: np.ndarray):
        super(SimulatedRRTSolver, self).__init__(sim, goal=goal)

    def solve(self) -> Union[np.ndarray, None]:
        ego_trajectory = np.expand_dims(self._env.ego.state, axis=0)

        for k in range(100):

            ado_trajectories = self._env.predict()

            return ego_trajectory

        return None
