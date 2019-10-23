from typing import List, Tuple, Union

import numpy as np

import murseco.planning.graph_search_x


def time_expanded_graph_search(
    pos_start: np.ndarray,
    pos_goal: np.ndarray,
    tppdf: List[np.ndarray],
    meshgrid: Tuple[np.ndarray, np.ndarray],
    risk_max: float,
    u_max: float = 1.0,
    u_resolution: float = 1.0,
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

    :argument pos_start: start position of agent (2,).
    :argument pos_goal: goal position of agent (2,).
    :argument tppdf: list of overall obstacle probability density functions (time-steps, x_grid, y_grid).
    :argument meshgrid: (x, y) coordinate meshgrid (which the tppdf is based on).
    :argument risk_max: maximal accumulated (i.e. summed over time) risk.
    :argument u_max: maximal L2-norm of input in one time-step [m].
    :argument u_resolution: resolution of input as L2-norm [m].
    :returns optimal trajectory and accumulated risk at every step, up to step t, (or None if no trajectory found)
    """
    assert u_max % u_resolution == 0, "u_max must be evenly dividable by its resolution"
    assert risk_max > 0, "accumulated risk must be positive"

    trajectory, risks_accumulated = murseco.planning.graph_search_x.time_expanded_graph_search(
        pos_start.flatten().astype(np.float32),
        pos_goal.flatten().astype(np.float32),
        np.array(tppdf).flatten().astype(np.float32),
        meshgrid[0].flatten().astype(np.float32),
        meshgrid[0].flatten().astype(np.float32),
        meshgrid[0].shape[1],
        meshgrid[1].shape[0],
        len(tppdf),
        risk_max,
        u_max,
        u_resolution,
    )

    if trajectory is None:
        return None, None
    else:
        risks_accumulated = [risks_accumulated[t] for t in range(len(risks_accumulated))]
        return np.array(trajectory), np.array(risks_accumulated)
