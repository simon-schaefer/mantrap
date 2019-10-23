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


cdef cont2discrete(float x, float y, list grid_data):
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


cdef coord2index(int ix, int iy, int y_size):
    return iy * y_size + ix


cdef coord3index(int ix, int iy, int t, int x_size, int y_size):
    return t * x_size * x_size + iy * x_size + ix


cdef dynamics(float x, float y, float ux, float uy):
    return x + ux, y + uy


cdef cost(float x, float y, float xg, float yg, float ux, float uy):
    return (x - xg) * (x - xg) + (y - yg) * (y - yg) + ux * ux + uy * uy


cdef class Node:

    cdef public float x
    cdef public float y
    cdef public int t
    cdef public float cost
    cdef public float pdf
    cdef public Node parent

    def __cinit__(self, float x, float y, float cost, float pdf, int time, Node parent):
        self.x = x
        self.y = y
        self.cost = cost
        self.pdf = pdf
        self.t = time
        self.parent = parent


cdef dijkstra_constrained(
    float x_start,
    float y_start,
    float x_goal,
    float y_goal,
    float[:] tppdf,
    float[:] x_grid,
    float[:] y_grid,
    int x_grid_dimension,
    int y_grid_dimension,
    int thorizon,
    float risk_max,
    float u_max,
    float u_resolution
):
    cdef int x_grid_size = x_grid_dimension
    cdef int y_grid_size = y_grid_dimension - 1

    cdef float x_grid_min = x_grid[<int>coord2index(0, 0, y_grid_size)]
    cdef float y_grid_min = y_grid[<int>coord2index(0, 0, y_grid_size)]
    cdef float x_grid_max = x_grid[<int>coord2index(y_grid_size, x_grid_size, y_grid_size)]
    cdef float y_grid_max = y_grid[<int>coord2index(y_grid_size, x_grid_size, y_grid_size)]
    cdef list grid_data = [x_grid_min, x_grid_max, x_grid_size, y_grid_min, y_grid_max, y_grid_size]

    # assert tppdf[<int>coord3index(5, 1, 10, x_grid_size, y_grid_size)] == tppdf[<int>coord3index(5, 1, 3, x_grid_size, y_grid_size)]

    cdef int u_delta = <int>(u_max / u_resolution)

    cdef int ix_start = -1
    cdef int iy_start = -1
    ix_start, iy_start = cont2discrete(x_start, y_start, grid_data)
    cdef int ix_goal = -1
    cdef int iy_goal = -1
    ix_goal, iy_goal = cont2discrete(x_goal, y_goal, grid_data)

    # Initialization - Build safe graph.
    cdef start_pdf = tppdf[<int>coord3index(ix_start, iy_start, 0, x_grid_size, y_grid_size)]
    cdef Node start_node = Node(x_start, y_start, 0.0, start_pdf, 0, None)
    cdef list queue = [start_node]
    cdef Node goal_node = None
    cdef float min_cost_goal = 9999999.9

    while len(queue) > 0:
        node = queue.pop(0)

        # for ux from 0 <= ux <= u_delta_x:
        for ux from +1 <= ux <= u_delta:
            for uy from -u_delta <= uy <= u_delta:
                x_next, y_next = dynamics(node.x, node.y, ux * u_resolution, uy * u_resolution)
                cost_next = node.cost + cost(node.x, node.y, x_goal, y_goal, ux * u_resolution , uy * u_resolution)
                ix_next, iy_next = cont2discrete(x_next, y_next, grid_data)
                pdf_next = node.pdf + tppdf[<int>coord3index(ix_next, iy_next, node.t, x_grid_size, y_grid_size)]
                node_next = Node(x_next, y_next, cost_next, pdf_next, node.t + 1, node)

                # Add only to neighbour if node within grid and constraints not violated.
                if ix_next >= 0 and iy_next >= 0:
                    if node_next.pdf < risk_max:  # pdf constraint

                        if ix_next == ix_goal and iy_next == iy_goal:
                            if node_next.cost < min_cost_goal:
                                goal_node = node_next
                                min_cost_goal = node_next.cost

                        if node_next.t < thorizon and node_next.cost < min_cost_goal:
                            queue.append(node_next)

        # print(node.x, node.y, node.pdf, node.t)

    # Get shortest path by recursively getting the parent node, starting from the goal node found.
    cdef list path = []
    cdef list pdfs = []
    if goal_node is not None:
        node = goal_node
        while node is not None:
            path.insert(0, (node.x, node.y))
            pdfs.insert(0, node.pdf)
            node = node.parent
        return path, pdfs

    else:
        return None, None


def time_expanded_graph_search(float[:] pos_start, float[:] pos_goal, float[:] tppdf, float[:] x_grid, float[:] y_grid,
                               int x_grid_dimension, int y_grid_dimension, int thorizon,
                               float risk_max, float u_max, float u_resolution
):
    trajectory, risk_sum = dijkstra_constrained(pos_start[0], pos_start[1], pos_goal[0], pos_goal[1], tppdf,
                                                x_grid, y_grid, x_grid_dimension, y_grid_dimension, thorizon,
                                                risk_max, u_max, u_resolution
    )
    return trajectory, risk_sum
