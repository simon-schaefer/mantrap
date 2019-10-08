from typing import Any, Dict, Tuple

import numpy as np

from murseco.obstacle.abstract import Obstacle
from murseco.utility.stats import Static2D


class StaticObstacle(Obstacle):
    """The StaticObstacle class represents a static and rigid obstacle, i.e. at each covered position it will
    be with probability 1.

    :argument borders: position/shape description in 2D space (x_min, x_max, y_min, y_max).
    """

    def __init__(self, borders: Tuple[float, float, float, float]):
        super(StaticObstacle, self).__init__("obstacle/static/StaticObstacle")
        assert len(borders) == 4, "borders tuple has to be in shape (x_min, x_max, y_min, y_max)"

        self.block_distribution = Static2D(np.array(borders))

    @property
    def pdf(self) -> Static2D:
        return self.block_distribution

    def summary(self) -> Dict[str, Any]:
        summary = super(StaticObstacle, self).summary()
        summary.update({"block": self.block_distribution.summary()})
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        super(StaticObstacle, cls).from_summary(json_text)
        block = Static2D.from_summary(json_text["block"])
        return cls((block.x_min, block.x_max, block.y_min, block.y_max))
