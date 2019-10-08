from abc import abstractmethod
from typing import Any

from murseco.utility.io import JSONSerializer


class Obstacle(JSONSerializer):
    def __init__(self, name):
        super(Obstacle, self).__init__(name)

    @abstractmethod
    def pdf(self) -> Any:
        """Return the probability density function of the obstacle at the current time."""
        pass

    def forward(self):
        """Forward simulate the obstacle, e.g. for discrete time obstacles increase time-step"""
        pass


class DiscreteTimeObstacle(Obstacle):
    def __init__(self, name: str, tmax: int):
        super(DiscreteTimeObstacle, self).__init__(name)
        assert tmax > 0, "discrete time horizon must be minimal 1"
        self.tmax = tmax  # maximal time-step.
        self.tn = 0  # time-step the model is in.

    def forward(self):
        assert self.tn + 1 < self.tmax, "maximal gmm-obstacle time-step trespassed"
        self.tn += 1

    @abstractmethod
    def pdf(self) -> Any:
        pass
