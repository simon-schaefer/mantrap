import abc
import math

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
