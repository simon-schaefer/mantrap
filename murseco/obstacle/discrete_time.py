from abc import abstractmethod
from typing import Any

from murseco.utility.io import JSONSerializer


class DiscreteTimeObstacle(JSONSerializer):

    def __init__(self, tmax: int):
        assert tmax > 0, "discrete time horizon must be minimal 1"
        self.tmax = tmax  # maximal time-step.

    @abstractmethod
    def pdf(self) -> Any:
        """Return the probability density function of the obstacle at the current time."""
        pass

    def forward(self):
        assert self.tn + 1 < self.tmax, "maximal gmm-obstacle time-step trespassed"
        self.tn += 1
