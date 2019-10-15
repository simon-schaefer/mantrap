from typing import Any, Dict, List, Tuple, Union

from murseco.obstacle.abstract import DTVObstacle
from murseco.robot.abstract import DTRobot
from murseco.utility.io import JSONSerializer


class Environment(JSONSerializer):
    def __init__(
        self,
        xaxis: Tuple[float, float],
        yaxis: Tuple[float, float],
        obstacles: List[DTVObstacle] = None,
        robot: DTRobot = None,
        **kwargs
    ):
        kwargs.update({"name": "environment/environment/Environment", "is_unique": False})
        super(Environment, self).__init__(**kwargs)

        assert xaxis[0] < xaxis[1], "xaxis has to be in the format (x_min, x_max)"
        assert yaxis[0] < yaxis[1], "yaxis has to be in the format (y_min, y_max)"

        self.xaxis = tuple(xaxis)
        self.yaxis = tuple(yaxis)
        self._obstacles = [] if obstacles is None else obstacles
        self._robot = robot
        self._tmax = 10

    def add_obstacle(self, obstacle: DTVObstacle):
        self._obstacles.append(obstacle)

    def add_robot(self, robot: DTRobot):
        assert self._robot is None, "just one robot possible in environment"
        self._tmax = robot.planning_horizon  # happens only once anyway that robot is added
        self._robot = robot

    @property
    def tmax(self) -> int:
        return self._tmax

    @property
    def obstacles(self) -> List[DTVObstacle]:
        return self._obstacles

    @property
    def robot(self) -> Union[None, DTRobot]:
        return self._robot

    def summary(self) -> Dict[str, Any]:
        summary = super(Environment, self).summary()
        obstacles = [o.summary() for o in self._obstacles]
        robot = self._robot.summary()
        summary.update({"xaxis": self.xaxis, "yaxis": self.yaxis, "obstacles": obstacles, "robot": robot})
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(Environment, cls).from_summary(json_text)
        obstacles = [cls.call_by_summary(obstacle_text) for obstacle_text in json_text["obstacles"]]
        robot = cls.call_by_summary(json_text["robot"])
        xaxis, yaxis = json_text["xaxis"], json_text["yaxis"]
        summary.update({"xaxis": xaxis, "yaxis": yaxis, "obstacles": obstacles, "robot": robot})
        return cls(**summary)
