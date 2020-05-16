import abc
import math
import typing

import numpy as np
import scipy.interpolate
import torch

import mantrap.utility.shaping


###########################################################################
# Numerical Methods #######################################################
###########################################################################
class Derivative2:
    """Determine the 2nd derivative of some series of point numerically by using the Central Difference Expression
    (with error of order dt^2). Assuming smoothness we can extract the acceleration from the positions:

    d^2/dt^2 x_i = \\frac{ x_{i + 1} - 2 x_i + x_{i - 1}}{ dt^2 }.

    For computational efficiency this expression, summed up over the full horizon (e.g. the full time-horizon
    [1, T - 1]), can be regarded as single matrix operation b = Ax with A = diag([1, -2, 1]).
    """

    def __init__(self, horizon: int, dt: float, num_axes: int = 2, velocity: bool = False):
        assert num_axes >= 2, "minimal number of axes of differentiable matrix is 2"
        self._num_axes = num_axes
        self._dt = dt

        if not velocity:  # acceleration estimate based on positions
            self._diff_mat = torch.zeros((horizon, horizon))
            for k in range(1, horizon - 1):
                self._diff_mat[k, k - 1] = 1
                self._diff_mat[k, k] = -2
                self._diff_mat[k, k + 1] = 1
            self._diff_mat *= 1 / dt ** 2

        else:  # acceleration estimate based on velocities
            self._diff_mat = torch.zeros((horizon, horizon))
            for k in range(1, horizon):
                self._diff_mat[k, k - 1] = -1
                self._diff_mat[k, k] = 1
            self._diff_mat *= 1 / dt

        # Un-squeeze difference matrix in case the state tensor is larger then two dimensional (batching).
        for _ in range(num_axes - 2):
            self._diff_mat = self._diff_mat.unsqueeze(0)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self._diff_mat, x)

    def compute_single(self, x: torch.Tensor, x_prev: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        return (x_prev - 2 * x + x_next) / self._dt**2


###########################################################################
# Interpolation ###########################################################
###########################################################################
def lagrange_interpolation(control_points: torch.Tensor, num_samples: int = 100, deg: int = 2) -> torch.Tensor:
    """Lagrange interpolation using Vandermonde Approach. Vandermonde finds the Lagrange parameters by solving
    a matrix equation X a = Y with known control point matrices (X,Y) and parameter vector a, therefore is fully
    differentiable. Also Lagrange interpolation guarantees to pass every control point, but performs poorly in
    extrapolation (which is however not required for trajectory fitting, since the trajectory starts and ends at
    defined control points.

    Source: http://www.maths.lth.se/na/courses/FMN050/media/material/lec8_9.pdf
    """
    assert len(control_points.shape) == 2
    assert control_points.shape[1] == 2

    x = torch.stack([control_points[:, 0] ** n for n in range(deg)], dim=1)
    y = control_points[:, 1]

    # If x is singular (i.e. det(x) = 0) the matrix is not invertible, give an error, since permeating some points
    # slightly to be able to perform an interpolation, leads to misinformed gradients. The result of the interpolation
    # won't be very stable (!).
    x = x + torch.eye(x.shape[0]) * 0.00001  # avoid singularities
    a = torch.inverse(x).matmul(y)

    x_up = torch.linspace(control_points[0, 0].item(), control_points[-1, 0].item(), steps=num_samples)
    y_up = torch.stack([x_up ** n for n in range(deg)]).t().matmul(a)
    return torch.stack((x_up, y_up), dim=1)


def spline_interpolation(control_points: torch.Tensor, num_samples: int = 100):
    assert mantrap.utility.shaping.check_ego_path(x=control_points)

    # B-Spline is not differentiable anyway.
    with torch.no_grad():
        control_points_np = control_points.detach().numpy()
        tck, _ = scipy.interpolate.splprep([control_points_np[:, 0], control_points_np[:, 1]], s=0.0)
        x_path, y_path = scipy.interpolate.splev(np.linspace(0, 1, num_samples), tck)
        path = torch.stack((torch.tensor(x_path), torch.tensor(y_path)), dim=1).float()

    assert mantrap.utility.shaping.check_ego_path(path, t_horizon=num_samples)
    return path


def grid_interpolation(values: np.ndarray, grid: typing.Tuple[np.ndarray, np.ndarray, np.ndarray], upscale: int
                       ) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Upscale an equally-spaced 2D grid using nearest neighbour.
    
    For up-scaling increase the number of steps within the grid dimensions, equally over all grid dimensions.
    Then using the scipy.interpolate framework upscale the values using the passed method.
    
    :param values: grid values to upscale (will be flattened).
    :param grid: grid description tuple (grid_min, grid_max, numer_of_points_by_dimension) - arrays.
    :param upscale: up-scaling factor.
    :returns: up_scaled values, new number of grid points by dimension
    """
    assert upscale >= 1
    grid_min, grid_max, n = grid

    # Create mesh-grid of original sized data from input grid data.
    grid_old = np.mgrid[grid_min[0]:grid_max[0]:(grid_max[0] - grid_min[0])/n[0],
                        grid_min[1]:grid_max[1]:(grid_max[1] - grid_min[1])/n[1],
                        grid_min[2]:grid_max[2]:(grid_max[2] - grid_min[2])/n[2],
                        grid_min[3]:grid_max[3]:(grid_max[3] - grid_min[3])/n[3]]

    # Create up-scaled mesh-grid by increasing the number of steps by the `upscale` - factor.
    n_new = upscale * n
    grid_new = np.mgrid[grid_min[0]:grid_max[0]:(grid_max[0] - grid_min[0])/n_new[0],
                        grid_min[1]:grid_max[1]:(grid_max[1] - grid_min[1])/n_new[1],
                        grid_min[2]:grid_max[2]:(grid_max[2] - grid_min[2])/n_new[2],
                        grid_min[3]:grid_max[3]:(grid_max[3] - grid_min[3])/n_new[3]]

    # Using the scipy.interpolate framework upscale the values from the old to the new representation.
    values_flat = values.flatten()
    grid_old = np.flip(np.rot90(np.transpose(grid_old)), 1).reshape(-1, 4)
    grid_new = np.flip(np.rot90(np.transpose(grid_new)), 1).reshape(-1, 4)

    values_up_scaled = scipy.interpolate.griddata(grid_old, values_flat, grid_new, method="nearest")
    values_up_scaled = values_up_scaled.reshape(*n_new.tolist())
    return values_up_scaled, n_new


###########################################################################
# Shapes ##################################################################
###########################################################################
class Shape2D(abc.ABC):

    @abc.abstractmethod
    def does_intersect(self, other: 'Shape2D') -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def samples(self, num_samples: int = 100) -> torch.Tensor:
        raise NotImplementedError


class Circle(Shape2D):
    def __init__(self, center: torch.Tensor, radius: float):
        assert mantrap.utility.shaping.check_2d_vector(center)
        assert radius >= 0.0  # 0 => zero velocity

        self.center = center.float()
        self.radius = float(radius)

    @classmethod
    def from_min_max_2d(cls, x_min, x_max, y_min, y_max):
        radius = (x_max - x_min) / 2
        center = torch.tensor([x_min + radius, y_min + radius])
        assert torch.isclose((y_max - y_min) / 2, radius, atol=0.01)
        return cls(center=center, radius=radius)

    def does_intersect(self, other: 'Circle') -> bool:
        if not other.__class__ == self.__class__:
            raise NotImplementedError

        distance = torch.norm(self.center - other.center)
        return distance < self.radius + other.radius

    def samples(self, num_samples: int = 100) -> torch.Tensor:
        angles = torch.linspace(start=0, end=2 * math.pi, steps=num_samples)
        dx_dy = torch.stack((torch.cos(angles), torch.sin(angles))).view(2, num_samples).t()
        return self.center + self.radius * dx_dy


###########################################################################
# Geometry ################################################################
###########################################################################
def straight_line(start: torch.Tensor, end: torch.Tensor, steps: int):
    """Create a 2D straight line from start to end position with `steps` number of points,
    with the first point being `start` and the last point being `end`. """
    line = torch.zeros((steps, 2))
    line[:, 0] = torch.linspace(start[0].item(), end[0].item(), steps)
    line[:, 1] = torch.linspace(start[1].item(), end[1].item(), steps)

    assert mantrap.utility.shaping.check_ego_path(line, t_horizon=steps)
    return line


def normal_line(start: torch.Tensor, end: torch.Tensor):
    """Normal direction to a line defined by  start and end position. """
    direction = (end - start) / torch.norm(end - start)
    return torch.tensor([direction[1], -direction[0]])
