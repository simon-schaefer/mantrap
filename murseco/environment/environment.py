from typing import Any, Dict, List, Tuple, Union

from murseco.obstacle.abstract import DiscreteTimeObstacle
from murseco.robot.abstract import DiscreteTimeRobot
from murseco.utility.types import EnvActor
from murseco.utility.io import JSONSerializer


class Environment(JSONSerializer):
    def __init__(self, xaxis: Tuple[float, float], yaxis: Tuple[float, float], actors: List[EnvActor] = None):
        super(Environment, self).__init__("environment/environment/Environment")

        assert xaxis[0] < xaxis[1], "xaxis has to be in the format (x_min, x_max)"
        assert yaxis[0] < yaxis[1], "yaxis has to be in the format (y_min, y_max)"

        self.xaxis = tuple(xaxis)
        self.yaxis = tuple(yaxis)
        self.actors = [] if actors is None else actors
        self._tmax = None

    def _add_actor(self, element: Any, category: str) -> str:
        """Add actor to list of actors and create its id from random.

        :argument element: actor object to add to list.
        :argument category: type of actor (obstacle, robot)
        :return id: obstacle assigned identifier string.
        """
        assert category in ("robot", "obstacle"), "agent type description must be one out of (robot, obstacle)"
        actor = EnvActor(category=category, element=element)
        self.actors.append(actor)
        return actor.identifier

    def add_obstacle(self, obstacle: DiscreteTimeObstacle) -> str:
        self._tmax = obstacle.tmax if self.tmax is None else self.tmax
        assert self._tmax == obstacle.tmax, "time horizon tmax should be equivalent for all obstacles"
        return self._add_actor(obstacle, "obstacle")

    def add_robot(self, robot: DiscreteTimeRobot) -> str:
        return self._add_actor(robot, "robot")

    @property
    def tmax(self) -> int:
        return self._tmax

    @property
    def obstacles(self) -> List[DiscreteTimeObstacle]:
        return [actor.element for actor in self.actors if actor.category == "obstacle"]

    @property
    def robot(self) -> Union[None, DiscreteTimeRobot]:
        robots = [actor for actor in self.actors if actor.category == "robot"]
        assert len(robots) < 2, "maximal one robot is allowed in the scene"
        return robots[0].element if len(robots) == 1 else None

    def summary(self) -> Dict[str, Any]:
        summary = super(Environment, self).summary()
        summary.update({"xaxis": self.xaxis, "yaxis": self.yaxis, "actors": [a.summary() for a in self.actors]})
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        actors = [EnvActor.from_summary(actor_text) for actor_text in json_text["actors"]]
        return cls(json_text["xaxis"], json_text["yaxis"], actors)
