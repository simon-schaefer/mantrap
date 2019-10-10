from abc import abstractmethod
from typing import Any, Dict, List

from murseco.utility.io import JSONSerializer
from murseco.utility.stats import Distribution2D


class DiscreteTimeObstacle(JSONSerializer):
    def __init__(self, name: str, tmax: int):
        super(DiscreteTimeObstacle, self).__init__(name)
        assert tmax > 0, "discrete time horizon must be minimal 1"
        self._tmax = tmax  # maximal time-step.

    @abstractmethod
    def pdf(self) -> Distribution2D:
        """Return the probability density function of the obstacle where to be at the next time step."""
        pass

    @abstractmethod
    def tpdf(self) -> List[Distribution2D]:
        """Return the probability density function of the obstacle over time."""
        pass

    @property
    def tmax(self) -> int:
        return self._tmax

    @abstractmethod
    def summary(self) -> Dict[str, Any]:
        summary = super(DiscreteTimeObstacle, self).summary()
        summary.update({"tmax": self._tmax})
        return summary
