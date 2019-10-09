from abc import abstractmethod
from typing import Any, Dict

from murseco.utility.io import JSONSerializer


class DiscreteTimeObstacle(JSONSerializer):
    def __init__(self, name: str, tmax: int):
        super(DiscreteTimeObstacle, self).__init__(name)
        assert tmax > 0, "discrete time horizon must be minimal 1"
        self.tmax = tmax  # maximal time-step.

    @abstractmethod
    def pdf(self) -> Any:
        """Return the probability density function of the obstacle at the current time."""
        pass

    @abstractmethod
    def tpdf(self) -> Any:
        """Return the probability density function of the obstacle over time."""
        pass

    @abstractmethod
    def summary(self) -> Dict[str, Any]:
        summary = super(DiscreteTimeObstacle, self).summary()
        summary.update({"tmax": self.tmax})
        return summary
