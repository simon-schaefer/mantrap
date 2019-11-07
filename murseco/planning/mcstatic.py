import heapq
import logging
from typing import Callable, List, Tuple, Union

import numpy as np

from murseco.problem import D2TSProblem


def _a_star_heuristic(vertex: np.ndarray, goal: np.ndarray) -> float:
    """Heuristic = L2 norm (euclidean distance between vertex and goal) since the environment does not have
    static obstacles and it always underestimates the real cost (consistent)"""
    return np.linalg.norm(vertex - goal)


def compute_safe_intervals(static_maps: np.ndarray) -> List[List[List[Tuple[float, float]]]]:
    """Compute safe intervals from static maps, a safe interval is defined as an interval in time in which the
    configuration (i.e. 2D position) is safe from collision with obstacles.

    :argument static_maps: boolean grid over time stating whether an obstacle is at (x,y) at time t (time-steps, x, y).
    :return safe intervals as list of tuples [(t0_start, t0_end), (t1_start, ....)] for all (x, y).
    """
    t_max = static_maps.shape[0]
    safe_intervals = [[[(0, t_max)] for _ in range(static_maps.shape[1])] for _ in range(static_maps.shape[2])]
    for ix in range(static_maps.shape[2]):
        for iy in range(static_maps.shape[1]):
            configuration = static_maps[:, iy, ix]

            if not np.sum(configuration) > 0:  # no collision in configuration
                continue

            changes = np.where(np.roll(configuration, 1) != configuration)[0]
            if configuration[0]:  # starting w/ obstacle
                safe_intervals[iy][ix] = [(changes[i], changes[i + 1]) for i in range(0, len(changes), 2)]
            else:  # starting w/o obstacle
                collisions = [(changes[i] - 2, changes[i + 1] - 2) for i in range(0, len(changes), 2)]
                safes = [(collisions[i - 1][1], collisions[i][0]) for i in range(1, len(collisions))]
                safes.insert(0, (0, collisions[0][0]))
                safes.insert(len(safes), (collisions[-1][1], t_max))
                safe_intervals[iy][ix] = safes

    return safe_intervals


def safe_interval_a_star(
    xi_start: np.ndarray,
    xi_goal: np.ndarray,
    static_maps: np.ndarray,
    dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray],
    action_space: np.ndarray,
    safe_intervals: List[List[List[Tuple[float, float]]]],
):
    """A* (AStar) path planning 3D space (2D position + time) for dynamic environments. To boost efficiency the
    "Safe Interval Path Planning" methodology is used, which basically reduces the in-fact dimensionality of the
    problem by regarding time-intervals instead of specific times. Therefore each "configuration" (2D position) is
    associated with collision-free "safe" intervals and intervals with collisions (False and True respectively in
    static_maps). However the algorithm guarantees to find the time-optimal path from start to goal node.

    A* with safe intervals
        g(s_start) = 0; OPEN = ∅;
        insert s_start into OPEN with f(s_start) = h(s_start);
        while(s_goal is not expanded)
            remove s with the smallest f-value from OPEN;
            successors = getSuccessors(s);
            for each s′ in successors
                if s′ was not visited before then
                    f(s′) = g(s′) = ∞;
                if g(s′) > g(s) + c(s, s′)
                    g(s′) = g(s) + c(s, s′); updateTime(s′);
                    f(s′) = g(s′) + h(s′);
                    insert s′ into OPEN with f(s′);

    getSuccessors(s)
        successors = ∅;
        for each m in M(s)
            cfg = configuration of m applied to s
            m_time = time to execute m
            start_t = time(s) + m_time
            end_t = endTime(interval(s)) + m_time
            for each safe interval i in cfg
                if startTime(i) > end_t or endTime(i) < start_t
                    continue
                t = earliest arrival time at cfg during interval i with no collisions
                if t does not exist
                    continue
                s′ = state of configuration cfg with interval i and time t insert s′ into successors
        return successors;


    :argument xi_start: start position in grid coordinates (x_i, y_i).
    :argument xi_goal: goal position in grid coordinates (x_i, y_i).
    :argument static_maps: dynamic environment (False=collision-free, True=some nonzero risk),(time-steps, x, y).
    :argument safe_intervals: safe intervals associated to every grid coordinate.
    :argument dynamics: robot dynamics mapping the current state and input to the next state.
    :argument action_space: robot's action space (iterable over possible actions).
    :return trajectory in grid coordinates (None if not existing).
    """

    cost_map = np.ones((static_maps.shape[2], static_maps.shape[1])) * np.inf
    seen_map = np.zeros((static_maps.shape[2], static_maps.shape[1]), dtype=bool)

    cost_map[xi_start[1], xi_start[0]] = 0
    i = 0
    # g_cost, time, index (for sorting in case of equivalent cost and time), vertex, trajectory
    heap = [(0 + _a_star_heuristic(xi_start, xi_goal), 0, i, xi_start, np.array([xi_start[0], xi_start[1], 0]))]
    while heap:
        (cost_i, ti, _, xi, trajectory) = heapq.heappop(heap)

        safe_interval_i = None
        for (t_min, t_max) in safe_intervals[xi[1]][xi[0]]:
            if t_min <= ti <= t_max:
                safe_interval_i = (t_min, t_max)
                break
        assert safe_interval_i is not None

        if not seen_map[xi[1], xi[0]]:
            seen_map[xi[1], xi[0]] = True
            trajectory = np.vstack((trajectory, np.array([xi[0], xi[1], ti])))
            if np.array_equal(xi, xi_goal):
                return trajectory[1:, :]

            successors = []
            actions = [(0, 1), (-1, 1), (0, -1), (1, -1), (1, 0), (-1, 1), (1, 0), (1, 1)]
            for m in actions:
                x_next = np.array(m) + xi
                if not 0 <= x_next[0] < cost_map.shape[1] or not 0 <= x_next[1] < cost_map.shape[0]:
                    continue
                m_time = np.ceil(np.linalg.norm(x_next - xi))
                t_start = ti + m_time
                t_end = safe_interval_i[1] + m_time

                for safe_interval in safe_intervals[x_next[1]][x_next[0]]:
                    if safe_interval[0] > t_end or safe_interval[1] < t_start:
                        continue
                    t_arrival = max(t_start, safe_interval[0])
                    successors.append((x_next, m_time, t_arrival))

            for xs, edge_cost_i_s, ts in successors:
                if seen_map[xs[1], xs[0]]:
                    continue
                s_cost = cost_i + edge_cost_i_s
                if s_cost < cost_map[xs[1], xs[0]]:
                    cost_map[xs[1], xs[0]] = s_cost
                    i = i + 1
                    heapq.heappush(heap, (s_cost + _a_star_heuristic(xs, xi_goal), ts, i, xs, trajectory))

    return None


def trajectory_flatten(trajectory: np.ndarray) -> Union[np.ndarray, None]:
    """Transform a 3D trajectory (x, y, t) to a 2D trajectory (x, y), each index for one time-step by repeating the
    same position (i.e. "wait") if trajectory[n].t == trajectory[n-1].t. Pass None trajectory.

    :argument trajectory: 3D trajectory (num_elements, 3).
    :return 2D trajectory (N, 2) with N >= num_elements.
    """
    if trajectory is None:
        return None

    trajectory_2d = trajectory[0, :2]
    for i in range(1, trajectory.shape[0]):
        x_prev, y_prev, t_prev = trajectory[i - 1, :]
        x, y, t = trajectory[i, :]
        assert t_prev < t, "invalid unsorted 3D trajectory"
        trajectory_2d = np.vstack((trajectory_2d, np.repeat(np.array([[x_prev, y_prev]]), t - t_prev - 1, axis=0)))
        trajectory_2d = np.vstack((trajectory_2d, np.array([x, y])))
    return trajectory_2d


def monte_carlo_static(
    problem: D2TSProblem, return_static_maps: bool = False
) -> Union[
    Tuple[Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None]],
    Tuple[Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None], np.ndarray],
]:

    logging.debug("starting D2TS monte_carlo_static planning")
    x_start, x_goal = problem.x_start_goal
    tppdf, _ = problem.grid

    logging.debug("starting static map transformation")
    risk_allocation = np.ones(problem.params.thorizon) * problem.params.risk_max
    static_maps = np.array(tppdf)
    static_maps[static_maps < risk_allocation] = 0
    static_maps[static_maps >= risk_allocation] = 1
    static_maps = static_maps.astype(bool)

    logging.debug("finding safe intervals")
    safe_intervals = compute_safe_intervals(static_maps)

    logging.debug("starting safe-interval astar")
    trajectory = safe_interval_a_star(
        xi_start=problem.cont2discrete(x_start[:2]),
        xi_goal=problem.cont2discrete(x_goal[:2]),
        static_maps=static_maps,
        dynamics=problem.robot_dynamics,
        action_space=problem.robot_action_space,
        safe_intervals=safe_intervals,
    )

    trajectory = trajectory_flatten(trajectory)  # (x, y, t) -> (x, y)_n
    trajectory = problem.discrete2cont(trajectory)  # (x_grid, y_grid) -> (x_cont, y_cont)

    if return_static_maps:
        return trajectory, None, None, static_maps
    else:
        return trajectory, None, None
