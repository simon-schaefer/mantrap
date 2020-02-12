from abc import abstractmethod

import numpy as np
import torch


###########################################################################
# Distributions ###########################################################
# Probability distribution classes for sampling ###########################
###########################################################################
class Distribution:
    def __init__(self, mean: float):
        self.mean = mean

    @abstractmethod
    def sample(self, num_samples: int = 1) -> np.ndarray:
        pass


class DirecDelta(Distribution):
    """Direc Delta distribution peaking at one specific point x, while have zero probability otherwise."""

    def __init__(self, x: float):
        super(DirecDelta, self).__init__(x)

    def sample(self, num_samples: int = 1) -> np.ndarray:
        super(DirecDelta, self).sample(num_samples)
        return np.tile(self.mean, num_samples).reshape(num_samples, -1)


class Gaussian(Distribution):
    """1D Gaussian distribution with mean mu and standard deviation sigma."""

    def __init__(self, mean: float, sigma: float):
        super(Gaussian, self).__init__(mean)
        self.sigma = sigma

    def sample(self, num_samples: int = 1) -> np.ndarray:
        super(Gaussian, self).sample(num_samples)
        return np.random.normal(self.mean, self.sigma, num_samples).reshape(num_samples, -1)


###########################################################################
# Numerical Methods #######################################################
###########################################################################
class Derivative2:
    """Determine the 2nd derivative of some series of point numerically by using the Central Difference Expression
    (with error of order dt^2). Assuming smoothness we can extract the acceleration from the positions:

    d^2/dt^2 x_i = \frac{ x_{i + 1} - 2 x_i + x_{i - 1}}{ dt^2 }.

    For computational efficiency this expression, summed up over the full horizon (e.g. the full time-horizon
    [1, T - 1]), can be regarded as single matrix operation b = Ax with A = diag([1, -2, 1]).
    """

    def __init__(self, horizon: int, dt: float, num_axes: int = 2):
        assert num_axes >= 2, "minimal number of axes of differentiable matrix is 2"
        self._num_axes = num_axes

        self._diff_mat = np.zeros((horizon, horizon))
        for k in range(1, horizon - 1):
            self._diff_mat[k, k - 1] = 1
            self._diff_mat[k, k] = -2
            self._diff_mat[k, k + 1] = 1
        self._diff_mat *= 1 / dt ** 2
        self._diff_mat = np.expand_dims(self._diff_mat, axis=tuple(i for i in range(num_axes - 2)))

    def compute(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(self._diff_mat, x)


###########################################################################
# Interpolation ###########################################################
###########################################################################
def lagrange_interpolation(control_points: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
    """Lagrange interpolation using Vandermonde Approach. Vandermonde finds the Lagrange parameters by solving
    a matrix equation X a = Y with known control point matrices (X,Y) and parameter vector a, therefore is fully
    differentiable. Also Lagrange interpolation guarantees to pass every control point, but performs poorly in
    extrapolation (which is however not required for trajectory fitting, since the trajectory starts and ends at
    defined control points.

    Source: http://www.maths.lth.se/na/courses/FMN050/media/material/lec8_9.pdf"""
    assert len(control_points.shape) == 2, "control points should be in shape (num_points, 2)"
    assert control_points.shape[1] == 2, "2D interpolation requires 2D input"
    n = control_points.shape[0]

    x = torch.stack((control_points[:, 0] ** 2, control_points[:, 0], torch.ones(n)), dim=1)
    y = control_points[:, 1]
    a = torch.inverse(x).matmul(y)

    x_up = torch.linspace(control_points[0, 0].item(), control_points[-1, 0].item(), steps=num_samples)
    y_up = torch.stack((x_up ** 2, x_up, torch.ones(num_samples))).T.matmul(a)
    return torch.stack((x_up, y_up), dim=1).float()


