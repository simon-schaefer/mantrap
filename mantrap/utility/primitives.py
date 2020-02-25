import numpy as np
import scipy.interpolate
import torch

from mantrap.constants import agent_speed_max
from mantrap.utility.shaping import check_ego_path


def square_primitives(start: torch.Tensor, end: torch.Tensor, dt: float, steps: int) -> torch.Tensor:
    """As the trajectory is optimized over several time-steps we have to set some base ego trajectory in order to
     estimate the reactive behaviour of all other agents. Therefore in the following some trajectory primitives
     between the current state of the ego and its goal state are defined, such as a first order (straight line from
     current to goal position) or higher order fits."""
    primitives = torch.ones((3, steps, 2)) * end.detach().numpy()

    distance = torch.norm(start - end).item()
    for i, normal_distance in enumerate([-distance / 2, 0, distance / 2]):
        primitives[i, :, :] = midpoint_spline(start, end, normal_distance, agent_speed_max * dt, steps)

    assert check_ego_path(primitives, t_horizon=steps, num_primitives=3)
    return primitives


def midpoint_spline(
    start_pos: torch.Tensor,
    end_pos: torch.Tensor,
    midpoint_distance: float,
    num_points: int,
    distance_per_step: float = 0.1,
    num_interpolation: int = 10,
) -> torch.Tensor:
    """B-Spline primitive interpolated between a start, an end position and a midpoint that is constructed by
    moving from the center point between start and end position in normal direction by the defined amount."""
    assert start_pos.size() == end_pos.size() == torch.Size([2]), "start and end position must be 2D positions"

    # Define fixed points of trajectory, like the starting point at the current ego position, the end point, which
    # is the goal point of optimization.
    start_point = start_pos.detach().numpy()
    end_point = end_pos.detach().numpy()
    distance = np.linalg.norm(end_point - start_point)

    # Different trajectories are obtained by varying its fixed points, starting from a direct straight line between
    # the start and end position the middle point is moved by going up- or downwards in the direction normal to
    # the direct connection.
    direction = (end_point - start_point) / distance
    mid_point = start_point + direction * distance / 2
    normal_direction = np.array([direction[1], -direction[0]])

    points = np.vstack((start_point, mid_point + normal_direction * midpoint_distance, end_point))
    psi = np.linspace(0, 1, num=points.shape[0])
    interpolator = scipy.interpolate.interp1d(psi, points, kind="quadratic", axis=0)
    path = interpolator(np.linspace(0, 1, 10))

    return torch.from_numpy(path)

    # # Re-sample path so that each point is equal-distant from its neighbours.
    # inter_distances = np.cumsum(np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1)))
    # inter_distances = np.asarray(np.hstack((np.zeros(1), inter_distances)) / distance_per_step, dtype=int)
    # _, inter_indices = np.unique(inter_distances, return_index=True)
    # if len(inter_indices) < num_points:
    #     inter_indices = np.hstack((inter_indices, np.ones(num_points - len(inter_indices)) * num_interpolation))
    # inter_indices[inter_indices >= num_interpolation] = num_interpolation - 1
    # primitive = path[inter_indices[:int(num_points)].astype(int), :]
    #
    # primitive = torch.from_numpy(primitive)
    # assert check_ego_path(primitive, t_horizon=num_points)
    # return primitive


def straight_line(start_pos: torch.Tensor, end_pos: torch.Tensor, steps: int):
    primitive = torch.zeros((steps, 2))
    primitive[:, 0] = torch.linspace(start_pos[0].item(), end_pos[0].item(), steps)
    primitive[:, 1] = torch.linspace(start_pos[1].item(), end_pos[1].item(), steps)
    assert check_ego_path(primitive, t_horizon=steps)
    return primitive
