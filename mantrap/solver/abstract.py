from abc import abstractmethod
from typing import Union

import numpy as np

from mantrap.simulation.abstract import Simulation


class Solver:
    def __init__(self, sim: Simulation, goal: np.ndarray):
        self._env = sim
        self._goal = goal

    @abstractmethod
    def solve(self) -> Union[np.ndarray, None]:
        """Solve the posed solver i.e. find a feasible path for the ego from its initial to its goal position/state.
        :return derived ego trajectory or None (no feasible solution)
        """
        pass

    @property
    def environment(self) -> Simulation:
        return self._env

    @property
    def goal(self) -> np.ndarray:
        return self._goal
