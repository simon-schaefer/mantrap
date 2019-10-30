import logging
from typing import Any, Callable, Tuple, Union

import numpy as np
import pandas as pd
import time

from murseco.problem import PROBLEMS
from murseco.utility.array import spline_re_sampling


COL_MIN_DISTANCE = "min_robot_obstacle_distance"
COL_RUNTIME_MS = "runtime"
COL_RISK_EMPIRICAL = "risk_empirical"
COL_RISK_CONSTRAINT = "risk_constraint"
COL_TRAVEL_TIME_S = "travel_time"


class Evaluation:
    def __init__(
        self,
        problem: Union[PROBLEMS],
        planner: Callable[[Union[PROBLEMS], Any], Tuple[np.ndarray, np.ndarray, np.ndarray]],
        test_risk: bool = True,
        test_travel_time: bool = True,
        test_runtime: bool = True,
        test_min_distance: bool = True,
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
        if test_runtime:
            tests.append(COL_RUNTIME_MS)
        if test_travel_time:
            tests.append(COL_TRAVEL_TIME_S)
        if test_min_distance:
            tests.append(COL_MIN_DISTANCE)
        self._results_df = pd.DataFrame(columns=tests)

        trajectory, controls, risks, runtime = self._solve(problem, planner, **planner_kwargs)
        for k in range(num_runs):
            logging.debug(f"evaluation: starting run {k}/{num_runs} ...")
            results = {}

            if not reuse_results:
                trajectory, controls, risks, runtime = self._solve(problem, planner, **planner_kwargs)

            if hasattr(problem, "params"):
                if hasattr(problem, "generate_trajectory_samples"):
                    obstacle_trajectories = problem.generate_trajectory_samples(num_samples=1).squeeze()
                    if test_risk:
                        results[COL_RISK_CONSTRAINT] = problem.params.risk_max
                        results[COL_RISK_EMPIRICAL] = int(self.check_for_collision(trajectory, obstacle_trajectories))
                    if test_min_distance:
                        results[COL_MIN_DISTANCE] = self.robot_obstacle_min_distance(trajectory, obstacle_trajectories)
                if test_travel_time and hasattr(problem, "x_start_goal"):
                    _, x_goal = problem.x_start_goal
                    results[COL_TRAVEL_TIME_S] = self.number_of_steps(trajectory, x_goal) * problem.params.dt
            if test_runtime:
                results[COL_RUNTIME_MS] = runtime
            self._results_df = self._results_df.append(results, ignore_index=True)

    @staticmethod
    def _solve(
        problem: Union[PROBLEMS],
        planner: Callable[[Union[PROBLEMS], Any], Tuple[np.ndarray, np.ndarray, np.ndarray]],
        **planner_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Running planner/solver for a given problem. However it is not possible to check against specific types
        aka that the type of problem (e.g. D2TS) the planner gets is actually the one the it is designed for (!).
        Therefore this has to be ensured by the user. """
        start_time = time.time()
        trajectory, controls, risks = planner(problem, **planner_kwargs)
        runtime = (time.time() - start_time) * 1000  # [ms]
        return trajectory, controls, risks, runtime

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
        robot_obstacle_distances = Evaluation.robot_obstacle_distances(robot_trajectory, obstacle_trajectories)
        num_collisions = np.sum(robot_obstacle_distances < max_collision_distance)
        return num_collisions > 0

    @staticmethod
    def robot_obstacle_distances(
        robot_trajectory: np.ndarray, obstacle_trajectories: np.ndarray, num_samples: int = 1000
    ) -> np.ndarray:
        """Determine the L2-distance between the robot and every obstacle trajectory. As described in the
        `check_for_collision`-method simply looking at the distance at every controlled time-step might overestimate
        the actual distance, since they could get closer in the time between two time-steps. Therefore re-sample
        the trajectories to a much denser resolution using B-splines (assuming to not violate the dynamics for
        reasonable small controlled dt) and return distances between the denser sampled trajectories.

        :argument robot_trajectory: trajectory of robot (time-horizon, n_state >= 2).
        :argument obstacle_trajectories: trajectory of every obstacle (num_obstacles, time-horizon, 2).
        :argument num_samples: number of sub-sampling samples.
        :return distances between robot and each obstacle (2, num_samples).
        """
        assert robot_trajectory.shape[0] == obstacle_trajectories.shape[1], "trajectories time-horizon must be equal"
        assert robot_trajectory.shape[1] == 2, "robot trajectories must be two-dimensional"
        assert obstacle_trajectories.shape[2] == 2, "obstacle trajectories must be two-dimensional"

        robot_trajectory_re = spline_re_sampling(robot_trajectory, num_sub_samples=num_samples)
        obstacle_trajectories_re = [spline_re_sampling(x, num_sub_samples=num_samples) for x in obstacle_trajectories]
        obstacle_trajectories_re = np.array(obstacle_trajectories_re)

        return np.linalg.norm(obstacle_trajectories_re[:, :, :] - robot_trajectory_re[:, :2], axis=2)

    @staticmethod
    def robot_obstacle_min_distance(robot_trajectory: np.ndarray, obstacle_trajectories: np.ndarray) -> float:
        robot_obstacle_distances = Evaluation.robot_obstacle_distances(robot_trajectory, obstacle_trajectories)
        return np.amin(robot_obstacle_distances)

    @staticmethod
    def number_of_steps(robot_trajectory: np.ndarray, x_goal: np.ndarray) -> Union[int, None]:
        """Count the number of time-steps until the robot reaches its goal state and (!) stays there until the end
        of the time-horizon. Otherwise return None.

        :argument robot_trajectory: trajectory of robot (time-horizon, n_state >= 2).
        :argument x_goal: robot's goal state (n_state >= 2).
        :return number of steps to reach the goal (None if not reached).
        """
        assert robot_trajectory.shape[1] == x_goal.shape[0], "trajectory and goal states must have same dimension"

        goal_reached = np.where(np.isclose(np.linalg.norm(robot_trajectory - x_goal, axis=1), 0, atol=0.1))[0]

        # Goal is only reached when the robot gets there and stays until the end of the time-horizon, therefore check
        # whether the last index of the trajectory is close to the goal. If not return None.
        if len(goal_reached) == 0 or goal_reached[-1] != robot_trajectory.shape[0] - 1:
            return None
        # Otherwise look for the smallest index for that the robot reaches the goal and stays there until the end of
        # the time-horizon and return it.
        else:
            goal_index = goal_reached[-1]
            while goal_index - 1 in goal_reached:
                goal_index = goal_index - 1
            return goal_index

    @property
    def results_df(self) -> pd.DataFrame:
        return self._results_df

    @property
    def results_mean(self) -> pd.DataFrame:
        return self._results_df.mean()

    @property
    def results_median(self) -> pd.DataFrame:
        return self._results_df.median()
