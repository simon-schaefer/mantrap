from abc import abstractmethod
from typing import Tuple, Union

import numpy as np

from mantrap.constants import planning_horizon_default
from mantrap.simulation.abstract import Simulation


class Solver:
    def __init__(self, sim: Simulation, goal: np.ndarray):
        self._env = sim
        self._goal = goal

    @abstractmethod
    def solve(self, t_horizon: int = planning_horizon_default) -> Tuple[Union[np.ndarray, None], np.ndarray]:
        """Solve the posed solver i.e. find a feasible trajectory for the ego from its initial to its goal state.
        :returns derived ego trajectory or None (no feasible solution) and according predicted ado trajectories
        """
        pass

    @property
    def environment(self) -> Simulation:
        return self._env

    @property
    def goal(self) -> np.ndarray:
        return self._goal
