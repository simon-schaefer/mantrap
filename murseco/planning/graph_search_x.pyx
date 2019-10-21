import numpy as np
cimport numpy as np

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
"""


cdef cont_2_discrete(float x, float y, list grid_data):
    x_min = grid_data[0]
    x_max = grid_data[1]
    x_size = <int>grid_data[2]
    y_min = grid_data[3]
    y_max = grid_data[4]
    y_size = <int>grid_data[5]

    if (not x_min <= x <= x_max) or (not y_min <= y <= y_max):
        return -1, -1

    cdef int ix = <int>((x - x_min) / (x_max - x_min) * (x_size - 1))  # start index at 0
    cdef int iy = <int>((y - y_min) / (y_max - y_min) * (y_size - 1))  # start index at 0
    return ix, iy


cdef dynamics(float x, float y, float ux, float uy):
    return x + ux, y + uy


cdef cost(float x, float y, float xg, float yg, float ux, float uy):
    return ux * ux + uy * uy


cdef heuristic(float x, float y, float xg, float yg):
    return (x - xg) * (x - xg) + (y - yg) * (y - yg)


cdef constraint(float pdf_sum, float pdf):
    return pdf_sum + pdf < 0.002


cdef class Node:

    cdef public float x
    cdef public float y
    cdef public int t
    cdef public float cost
    cdef public float pdf

    def __cinit__(self, float x, float y, float cost, float pdf, int time):
        self.x = x
        self.y = y
        self.cost = cost
        self.pdf = pdf
        self.t = time

cdef dijkstra_constrained(
        float x_start, float y_start, float x_goal, float y_goal, list tppdf, tuple meshgrid, float u_max
):
    cdef int x_grid_size = meshgrid[0].shape[1]
    cdef int y_grid_size = meshgrid[1].shape[1]
    cdef float x_grid_min = meshgrid[0][0, 0]
    cdef float y_grid_min = meshgrid[1][0, 0]
    cdef float x_grid_max = meshgrid[0][0, -1]
    cdef float y_grid_max = meshgrid[1][-1, 0]
    cdef list grid_data = [x_grid_min, x_grid_max, x_grid_size, y_grid_min, y_grid_max, y_grid_size]

    cdef int u_delta_x = int(u_max / ((meshgrid[0][0, -1] - meshgrid[0][0, 0]) / x_grid_size))
    cdef int u_delta_y = int(u_max / ((meshgrid[1][-1, 0] - meshgrid[1][0, 0]) / y_grid_size))
    cdef int thorizon = len(tppdf) + 1

    cdef int ix_start = -1
    cdef int iy_start = -1
    ix_start, iy_start = cont_2_discrete(x_start, y_start, grid_data)
    cdef int ix_goal = -1
    cdef int iy_goal = -1
    ix_goal, iy_goal = cont_2_discrete(x_goal, y_goal, grid_data)

    cdef list queue = []
    cdef list path = []
    cdef float pdf_sum = 0
    cdef int ix = -1
    cdef int iy = -1
    ix, iy = cont_2_discrete(x_start, y_start, grid_data)

    # Initialization.
    node = Node(x_start, y_start, 0.0, tppdf[0][ix, iy], 0)
    queue.append(node)

    # Loop.
    while len(queue) > 0:
        node = queue.pop(0)
        # if not constraint(pdf_sum, node.pdf):
        #     continue

        ix, iy = cont_2_discrete(node.x, node.y, grid_data)
        path.append((node.x, node.y))
        pdf_sum = pdf_sum + node.pdf

        if ix == ix_goal and iy == iy_goal:
            return path, pdf_sum

        for ux from -u_delta_x <= ux <= u_delta_x:
            for uy from -u_delta_y <= uy <= u_delta_y:
                x_next, y_next = dynamics(node.x, node.y, ux, uy)
                cost_next = node.cost + cost(node.x, node.y, x_goal, y_goal, ux, uy)
                ix_next, iy_next = cont_2_discrete(x_next, y_next, grid_data)
                node_next = Node(x_next, y_next, cost_next, tppdf[node.t + 1][ix_next, iy_start], node.t + 1)

                if ix_next >= 0 and iy_next >= 0:  # node within grid
                    # if constraint(pdf_sum, node_next.pdf):  # constraint not violated
                    i = 0
                    while i < len(queue) and \
                        cost_next + heuristic(node_next.x, node_next.y, x_goal, y_goal) > \
                        queue[i].cost + heuristic(queue[i].x, queue[i].y, x_goal, y_goal):
                        i = i + 1
                    queue.insert(i, node_next)

    return None, pdf_sum



def time_expanded_graph_search(np.ndarray pos_start, np.ndarray pos_goal,list tppdf, tuple meshgrid, float u_max = 3.0):
    trajectory, risk_sum = dijkstra_constrained(
        pos_start[0], pos_start[1], pos_goal[0], pos_goal[1], tppdf, meshgrid, u_max
    )
    trajectory = trajectory + [pos_goal] * (len(tppdf) - len(trajectory) + 1)
    return np.array(trajectory), risk_sum
