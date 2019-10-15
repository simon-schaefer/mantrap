from typing import Any, Dict, List, Tuple, Union

import numpy as np

from murseco.obstacle.abstract import DTVObstacle
from murseco.robot.abstract import DTRobot
from murseco.utility.io import JSONSerializer


class Environment(JSONSerializer):
    def __init__(
        self,
        xaxis: Tuple[float, float],
        yaxis: Tuple[float, float],
        dt: float = 1.0,
        thorizon: int = 10,
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
        self._thorizon = thorizon
        self._dt = dt

    def tppdf(self, num_points: int = 300) -> Tuple[List[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Determine pdf of obstacles in position space over full (discrete) time-horizon.
        Therefore at first determine for each obstacle the pdf in position space (ppdf) over the full time horizon
        and afterwards for each time-step sum the pdf over all obstacles to one overall pdf. While the ppdf of
        each obstacle and time-step is stored at "general" distribution (i.e. Gaussian2D) at first, for summation
        the pdf is evaluated at each point of a grid (2D numpy linspace).

        :argument num_points: number of resolution points for grid sampling for each axis.
        :returns overall pdf in position space for each time-step (list), (x, y) meshgrid
        """
        tppdfs = [o.ppdf(self.thorizon) for o in self.obstacles]

        num_points = int(num_points)
        x_min, x_max, y_min, y_max = self.xaxis[0], self.xaxis[1], self.yaxis[0], self.yaxis[1]
        x, y = np.meshgrid(np.linspace(x_min, x_max, num_points), np.linspace(y_min, y_max, num_points))

        tppdfs_overall = [sum([tppdfs[i][t].pdf_at(x, y) for i in range(len(tppdfs))]) for t in range(self.thorizon)]
        return tppdfs_overall, (x, y)

    def generate_trajectory_samples(self, num_samples_per_mode: int = 5) -> List[np.ndarray]:
        """Generate trajectory samples for each obstacle in the environment, split by the obstacle's mode.
        Therefore for each obstacle create N = num_samples_per_mode trajectories in each mode, by calling the obstacles
        `trajectory_samples` with some mode, ending up in an array (num_modes, N, time-horizon, 2) per obstacle.

        :argument num_samples_per_mode: number of trajectories to sample (per mode).
        :returns trajectories for every mode for every obstacle in the environment
        """
        samples = []
        for obstacle in self.obstacles:
            obstacle_samples = np.zeros((obstacle.num_modes, num_samples_per_mode, self.thorizon, 2))
            for m in range(obstacle.num_modes):
                mode_samples = obstacle.trajectory_samples(self.thorizon, num_samples=num_samples_per_mode, mode=m)
                obstacle_samples[m, :, :, :] = mode_samples
            samples.append(obstacle_samples)
        return samples

    def add_obstacle(self, obstacle_object, **obstacle_kwargs):
        obstacle = obstacle_object(dt=self._dt, **obstacle_kwargs)
        self._obstacles.append(obstacle)

    def add_robot(self, robot_object, **robot_kwargs):
        assert self._robot is None, "just one robot possible in environment"
        self._robot = robot_object(thorizon=self._thorizon, **robot_kwargs)

    @property
    def thorizon(self) -> int:
        return self._thorizon

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
