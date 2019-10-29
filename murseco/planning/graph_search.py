from typing import Tuple, Union

import logging
import numpy as np

from murseco.problem import D2TSProblem
import murseco.planning.graph_search_x


def time_expanded_graph_search(
    problem: D2TSProblem, u_resolution: float = 1.0
) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
    """Create time-expanded graph from temporal-position-based pdf as well as costs function and search for shortest
    path from initial to final position.
    A time-expanded graph consists of sub-graphs, one graph for each time step (time horizon is inferred from the
    length of the tppdf), so each grid cell is associated with one node in each sub-graph. Edges can merely go
    forward in time. In order to reduce the number of edges, for scalability, the input is constrained to be smaller
    than u_max (L2-norm).
    For shortest path search the Dijkstra algorithm is used. However storing all costs and node status in array would not
    be scalable to a large number of nodes (e.g. for a 100x100 grid, 20 time-steps the usually utilised adjacency
    representation of the graph would use roughly 300 GB of memory, using floats). Therefore it is exploited that the
    edges can only go forward in time, so that the resulting graph is in fact a tree structure, i.e. the leafs that are
    expanded from node n1(t = t*) do not have any edges to another node n2(t = t*). Nodes can therefore not be revisited,
    therefore there is no need to store a list containing the node's status (!).

    :argument problem: temporally and spatial discrete problem formulation.
    :argument u_resolution: resolution of input as L2-norm [m].
    :returns optimal trajectory and accumulated risk at every step, up to step t, (or None if no trajectory found)
    """
    assert problem.params.u_max % u_resolution == 0, "u_max must be evenly dividable by its resolution"
    assert problem.params.risk_max > 0, "accumulated risk must be positive"

    x_start, x_goal = problem.x_start_goal
    tppdf, (x_grid, y_grid) = problem.grid

    logging.debug("starting D2TS time-expanded graph search")
    trajectory, risks_accumulated = murseco.planning.graph_search_x.time_expanded_graph_search(
        x_start.flatten().astype(np.float32),
        x_goal.flatten().astype(np.float32),
        np.array(tppdf).flatten().astype(np.float32),
        x_grid.flatten().astype(np.float32),
        y_grid.flatten().astype(np.float32),
        x_grid.shape[1],
        y_grid.shape[0],
        problem.params.thorizon,
        problem.params.risk_max,
        problem.params.u_max,
        problem.params.w_x,
        problem.params.w_u,
        u_resolution,
    )
    logging.debug(f"found D2TS time-expanded graph search solution ? {trajectory is not None}")

    if trajectory is None:
        return None, None
    else:
        trajectory = np.array(trajectory)
        risks_accumulated = np.array([risks_accumulated[t] for t in range(len(risks_accumulated))])

        # If trajectory is shorter then problem's time horizon, then repetitively append goal point and last
        # accumulated risk to arrays.
        num_to_add = problem.params.thorizon - trajectory.shape[0]
        trajectory_additional = np.reshape(np.array([x_goal] * num_to_add), (num_to_add, 2))
        trajectory = np.vstack((trajectory, trajectory_additional))
        risks_accumulated_additional = np.array([risks_accumulated[-1]] * num_to_add)
        risks_accumulated = np.hstack((risks_accumulated, risks_accumulated_additional))

        return trajectory, risks_accumulated
