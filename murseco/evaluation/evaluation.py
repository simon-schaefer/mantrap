import logging
from typing import Any, Callable, Tuple, Union

import numpy as np
import pandas as pd
import time

from murseco.problem import PROBLEMS
from murseco.utility.array import spline_re_sampling


COL_RUNTIME_MS = "runtime"
COL_RISK_EMPIRICAL = "risk_empirical"
COL_RISK_CONSTRAINT = "risk_constraint"


class Evaluation:
    def __init__(
        self,
        problem: Union[PROBLEMS],
        planner: Callable[[Union[PROBLEMS], Any], Tuple[np.ndarray, np.ndarray]],
        test_risk: bool = True,
        test_time: bool = True,
        reuse_results: bool = False,
        num_runs: int = 100,
        **planner_kwargs,
    ):
        """Evaluation of performance of some planner for a specific problem. Since the problems are stochastic
        a single test run would not be sufficient to reflect the behaviour of the planner for the given problem.
        Therefore it is evaluated in a Monte-Carlo-manner, i.e. by repeatedly sampling trajectories of the probability
        distribution of the obstacles while applying the determined policy/trajectory for the robot.
        """
        tests = []
        if test_risk:
            tests.append(COL_RISK_EMPIRICAL)
            tests.append(COL_RISK_CONSTRAINT)
        if test_time:
            tests.append(COL_RUNTIME_MS)
        self._results_df = pd.DataFrame(columns=tests)

        trajectory, risks, runtime = self._solve(problem, planner, **planner_kwargs)
        for k in range(num_runs):
            logging.debug(f"evaluation: starting run {k}/{num_runs} ...")
            results = {}

            if not reuse_results:
                trajectory, risks, runtime = self._solve(problem, planner, **planner_kwargs)

            if test_risk and hasattr(problem, "params") and hasattr(problem, "generate_trajectory_samples"):
                obstacle_trajectories = problem.generate_trajectory_samples(num_samples=1).squeeze()
                results[COL_RISK_CONSTRAINT] = problem.params.risk_max
                results[COL_RISK_EMPIRICAL] = int(self.check_for_collision(trajectory, obstacle_trajectories))
            if test_time:
                results[COL_RUNTIME_MS] = runtime
            self._results_df = self._results_df.append(results, ignore_index=True)

    @staticmethod
    def _solve(
        problem: Union[PROBLEMS],
        planner: Callable[[Union[PROBLEMS], Any], Tuple[np.ndarray, np.ndarray]],
        **planner_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Running planner/solver for a given problem. However it is not possible to check against specific types
        aka that the type of problem (e.g. D2TS) the planner gets is actually the one the it is designed for (!).
        Therefore this has to be ensured by the user. """
        start_time = time.time()
        trajectory, risks = planner(problem, **planner_kwargs)
        runtime = (time.time() - start_time) * 1000  # [ms]
        return trajectory, risks, runtime

    @staticmethod
    def check_for_collision(
        robot_trajectory: np.ndarray, obstacle_trajectories: np.ndarray, max_collision_distance: float = 0.01
    ) -> bool:
        """Check for collision between robot trajectory and trajectory of each obstacle. Thereby a collision is
        defined as the robot approaching an obstacle closer than the max_collision_distance, at any point of time.
        As described in the paper "Monte Carlo Motion Planning for Robot Trajectory Optimization Under Uncertainty"
        by Lucas Janson, Edward Schmerling, and Marco Pavone in order to determine the real collision probability
        of some trajectory in an environment it usually is not sufficient to evaluate discrete points of this
        trajectory, since the inter-point area might be very close to an obstacle. Therefore the trajectory is
        interpolated using B-splines and sampled in large frequency. The resulting dense trajectory is then taken
        into account for determining the actual risk of collision.

        :argument robot_trajectory: trajectory of robot (time-horizon, n_state >= 2).
        :argument obstacle_trajectories: trajectory of every obstacle (num_obstacles, time-horizon, 2).
        :argument max_collision_distance: minimal L2 distance between robot and obstacle to be collision-free.
        :return flag indicating whether there is a collision happening (True) or not (False)
        """
        assert robot_trajectory.shape[0] == obstacle_trajectories.shape[1], "trajectories time-horizon must be equal"
        assert robot_trajectory.shape[1] == 2, "robot trajectories must be two-dimensional"
        assert obstacle_trajectories.shape[2] == 2, "obstacle trajectories must be two-dimensional"

        robot_trajectory_re = spline_re_sampling(robot_trajectory, num_sub_samples=1000)
        obstacle_trajectories_re = [spline_re_sampling(x, num_sub_samples=1000) for x in obstacle_trajectories]
        obstacle_trajectories_re = np.array(obstacle_trajectories_re)

        interstate_distances = np.linalg.norm(obstacle_trajectories_re[:, :, :] - robot_trajectory_re[:, :2], axis=2)
        num_collisions = np.sum(interstate_distances < max_collision_distance)
        return num_collisions > 0

    @property
    def results_df(self) -> pd.DataFrame:
        return self._results_df

    @property
    def results_mean(self) -> pd.DataFrame:
        return self._results_df.mean()

    @property
    def results_median(self) -> pd.DataFrame:
        return self._results_df.median()
