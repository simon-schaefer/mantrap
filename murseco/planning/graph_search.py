from typing import Callable, List, Tuple, Union

import igraph
import numpy as np
import re

from murseco.utility.graph import grid_cont_2_discrete


def time_expanded_graph_search(
    pos_start: np.ndarray,
    pos_goal: np.ndarray,
    tppdf: List[np.ndarray],
    costs: Callable[[np.ndarray, np.ndarray], float],
    dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray],
    meshgrid: Tuple[np.ndarray, np.ndarray],
    u_max: float = 3.0,
) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None], igraph.Graph]:
    """Create time-expanded graph from temporal-position-based pdf as well as costs function and search for shortest
    path from initial to final position.
    A time-expanded graph consists of sub-graphs, one graph for each time step (time horizon is inferred from the
    length of the tppdf), so each grid cell is associated with one node in each sub-graph. Edges can merely go
    forward in time. In order to reduce the number of edges, for scalability, the input is constrained to be smaller
    than u_max (L2-norm).
    For path search apply dijkstra's algorithm, which is surprisingly efficient implemented in the igraph  package,
    also for large graphs with several million edges it converges after a few hundred milliseconds
    (https://www.timlrx.com/2019/05/05/benchmark-of-popular-graph-network-packages/).

    :argument pos_start: two-dimensional initial position of agent (2,).
    :argument pos_goal: two-dimensional goal position of agent (2,).
    :argument tppdf: temporal-position-based pdf (thorizon, n_y, n_x).
    :argument costs: transition cost function for applying u_k in state x_k, i.e. l(x_k, u_k).
    :argument dynamics: system dynamics function f such that x_t+1 = f(x_k, u_k).
    :argument meshgrid: numpy meshgrid (x, y) (thorizon, n_y, n_x).
    :argument u_max: maximal input norm for possible state transitions.
    :returns trajectory as array (num_path_points, 2), risk allocation as array (num_path_points,), igraph graph.
    """
    x_grid_size, y_grid_size = meshgrid[0].shape[1], meshgrid[1].shape[0]
    u_delta_x = int(u_max / ((meshgrid[0][0, -1] - meshgrid[0][0, 0]) / x_grid_size))
    u_delta_y = int(u_max / ((meshgrid[1][-1, 0] - meshgrid[1][0, 0]) / y_grid_size))
    thorizon = len(tppdf) + 1
    ipos_start = grid_cont_2_discrete(pos_start, meshgrid)
    ipos_goal = grid_cont_2_discrete(pos_goal, meshgrid)

    # Build graph with grid cells as vertices, state transitions as edges and state- and input-dependent costs
    # as edge weight.
    # TODO: Actually building the graph is by far the slowest part of the algorithm !!
    def _name(x_coord: int, y_coord: int, t_coord: int) -> str:
        return f"x={x_coord}|y={y_coord}|t={t_coord}"

    # Create graph vertices of all times for full grid.
    graph = igraph.Graph(directed=True)
    for t in range(thorizon):
        for ix in range(x_grid_size):
            for iy in range(y_grid_size):
                vname = _name(ix, iy, t)
                ppdf = tppdf[t - 1][ix, iy] if t > 0 else 0.0
                graph.add_vertex(name=vname, ppdf=ppdf)

    # Iteratively add edges in feasible space.
    edges = []
    weights = []
    for t in range(thorizon - 1):
        for ix in range(x_grid_size):
            for iy in range(y_grid_size):
                x, y = meshgrid[0][iy, ix], meshgrid[1][iy, ix]
                for ux in range(-u_delta_x, u_delta_x):
                    for uy in range(-u_delta_y, u_delta_y):
                        x_vector, u_vector = np.array([x, y]), np.array([ux, uy])
                        pos_next = dynamics(x_vector, u_vector)
                        ix_next, iy_next = grid_cont_2_discrete(pos_next, meshgrid)
                        if ix_next is None or iy_next is None:
                            continue
                        cost = costs(x_vector, u_vector)
                        vname, vname_next = _name(ix, iy, t), _name(ix_next, iy_next, t + 1)
                        edges.append((vname, vname_next))
                        weights.append(cost)
    graph.add_edges(edges)

    # Find the shortest path using Dijkstra's algorithm.
    ipos_start_vertex = _name(ipos_start[0], ipos_start[1], 0)
    ipos_goal_vertex = _name(ipos_goal[0], ipos_goal[1], thorizon - 1)
    path_vertices = graph.get_shortest_paths(ipos_start_vertex, ipos_goal_vertex, weights, output="vpath")[0]

    if len(path_vertices) == 0:
        return None, None, graph
    else:
        trajectory = np.zeros((thorizon, 2))
        risks = np.zeros(thorizon)
        for k, vertex in enumerate(path_vertices):
            ix, iy, t = [int(k) for k in re.findall(r"\d+", graph.vs[vertex]["name"])]
            assert t == k, "time must always go forwards by one step in path"
            trajectory[k, :] = np.array([meshgrid[0][iy, ix], meshgrid[1][iy, ix]])
            risks[k] = tppdf[k - 1][ix, iy] if k > 0 else 0
        return trajectory, risks, graph
