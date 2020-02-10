import numpy as np
import scipy.interpolate
import torch

from mantrap.agents.agent import Agent
from mantrap.constants import agent_speed_max, solver_horizon
from mantrap.utility.shaping import check_trajectory_primitives


def square_primitives(agent: Agent, goal: torch.Tensor, dt: float, num_points: int = solver_horizon) -> torch.Tensor:
    """As the trajectory is optimized over several time-steps we have to set some base ego trajectory in order to
     estimate the reactive behaviour of all other agents. Therefore in the following some trajectory primitives
     between the current state of the ego and its goal state are defined, such as a first order (straight line from
     current to goal position) or higher order fits."""
    primitives = np.ones((3, num_points, 2)) * goal.detach().numpy()
    distance_per_step = agent_speed_max * dt
    num_interp = 1000

    # Define fixed points of trajectory, like the starting point at the current ego position, the end point, which
    # is the goal point of optimization.
    start_point = agent.position.detach().numpy()
    end_point = goal.detach().numpy()
    distance = np.linalg.norm(end_point - start_point)

    # Different trajectories are obtained by varying its fixed points, starting from a direct straight line between
    # the start and end position the middle point is moved by going up- or downwards in the direction normal to
    # the direct connection.
    direction = (end_point - start_point) / distance
    mid_point = start_point + direction * distance / 2
    normal_direction = np.array([direction[1], -direction[0]])
    for i, normal_distance in enumerate([-distance / 2, 0, distance / 2]):
        points = np.vstack((start_point, mid_point + normal_direction * normal_distance, end_point))
        psi = np.linspace(0, 1, num=points.shape[0])
        interpolator = scipy.interpolate.interp1d(psi, points, kind="quadratic", axis=0)
        path = interpolator(np.linspace(0, 1, num_interp))

        # Re-sample path so that each point is equal-distant from its neighbours.
        inter_distances = np.cumsum(np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1)))
        inter_distances = np.asarray(np.hstack((np.zeros(1), inter_distances)) / distance_per_step, dtype=int)
        _, inter_indices = np.unique(inter_distances, return_index=True)
        if len(inter_indices) < num_points:
            inter_indices = np.hstack((inter_indices, np.ones(num_points - len(inter_indices)) * num_interp))
        inter_indices[inter_indices >= num_interp] = num_interp - 1
        primitives[i, :, :] = path[inter_indices[:num_points].astype(int), :]

    primitives = torch.from_numpy(primitives)
    assert check_trajectory_primitives(primitives, t_horizon=num_points, num_primitives=3)
    return primitives


def straight_line_primitive(horizon: int, start_pos: torch.Tensor, end_pos: torch.Tensor):
    primitive = torch.zeros((horizon, 2))
    primitive[:, 0] = torch.linspace(start_pos[0].item(), end_pos[0].item(), horizon)
    primitive[:, 1] = torch.linspace(start_pos[1].item(), end_pos[1].item(), horizon)
    assert check_trajectory_primitives(primitive, t_horizon=horizon)
    return primitive
