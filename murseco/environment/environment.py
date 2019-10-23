import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from murseco.obstacle.abstract import DTVObstacle
from murseco.robot.abstract import DTRobot
from murseco.utility.io import JSONSerializer


class Environment(JSONSerializer):
    def __init__(
        self,
        xaxis: Tuple[float, float] = (-10, 10),
        yaxis: Tuple[float, float] = (-10, 10),
        dt: float = 1.0,
        obstacles: List[DTVObstacle] = None,
        robot: DTRobot = None,
        **kwargs,
    ):
        kwargs.update({"name": "environment/environment/Environment", "is_unique": False})
        super(Environment, self).__init__(**kwargs)

        assert xaxis[0] < xaxis[1], "xaxis has to be in the format (x_min, x_max)"
        assert yaxis[0] < yaxis[1], "yaxis has to be in the format (y_min, y_max)"

        self.xaxis = tuple(xaxis)
        self.yaxis = tuple(yaxis)
        self._obstacles = [] if obstacles is None else obstacles
        self._robot = robot
        self._dt = dt

    def tppdf(
        self, thorizon: int = 20, num_points: int = 1000, mproc: bool = True
    ) -> Tuple[List[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Determine pdf of obstacles in position space over full (discrete) time-horizon.
        Therefore at first determine for each obstacle the pdf in position space (ppdf) over the full time horizon
        and afterwards for each time-step sum the pdf over all obstacles to one overall pdf. While the ppdf of
        each obstacle and time-step is stored at "general" distribution (i.e. Gaussian2D) at first, for summation
        the pdf is evaluated at each point of a grid (2D numpy linspace).

        :argument thorizon: number of time-steps to forward simulate i.e. number of ppdf instances.
        :argument num_points: number of resolution points for grid sampling for each axis.
        :argument mproc: run in multiprocessing (8 processes).
        :returns overall pdf in position space for each time-step in meshgrid, (x, y) meshgrid
        """
        tppdfs = [o.tppdf(thorizon, mproc=mproc) for o in self.obstacles]
        logging.debug(f"{self.name}: determined analytical tppdf distribution")

        num_points = int(num_points)
        x_min, x_max, y_min, y_max = self.xaxis[0], self.xaxis[1], self.yaxis[0], self.yaxis[1]
        x, y = np.meshgrid(np.linspace(x_min, x_max, num_points), np.linspace(y_min, y_max, num_points))

        tppdfs_overall = [sum([tppdfs[i][t].pdf_at(x, y) for i in range(len(tppdfs))]) for t in range(thorizon)]
        logging.debug(f"{self.name}: determined grid tppdf distribution")
        return tppdfs_overall, (x, y)

    def generate_trajectory_samples(
        self, thorizon: int = 20, num_samples_per_mode: int = 5, mproc: bool = True
    ) -> List[np.ndarray]:
        """Generate trajectory samples for each obstacle in the environment, split by the obstacle's mode.
        Therefore for each obstacle create N = num_samples_per_mode trajectories in each mode, by calling the obstacles
        `trajectory_samples` with some mode, ending up in an array (num_modes, N, time-horizon, 2) per obstacle.

        :argument thorizon: number of time-steps i.e. length of trajectory.
        :argument num_samples_per_mode: number of trajectories to sample (per mode).
        :argument mproc: run in multiprocessing (8 processes).
        :returns trajectories for every mode for every obstacle in the environment
        """
        samples = []
        for obstacle in self.obstacles:
            obstacle_samples = np.zeros((obstacle.num_modes, num_samples_per_mode, thorizon, 2))
            for m in range(obstacle.num_modes):
                mode_samples = obstacle.trajectory_samples(thorizon, num_samples_per_mode, mode=m, mproc=mproc)
                obstacle_samples[m, :, :, :] = mode_samples
            samples.append(obstacle_samples)
        logging.debug(f"{self.name}: generated {len(samples)} trajectory samples")
        return samples

    def add_obstacle(self, obstacle_object, **obstacle_kwargs):
        obstacle = obstacle_object(dt=self._dt, **obstacle_kwargs)
        self._obstacles.append(obstacle)

    def add_robot(self, robot_object, **robot_kwargs):
        assert self._robot is None, "just one robot possible in environment"
        self._robot = robot_object(**robot_kwargs)

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
