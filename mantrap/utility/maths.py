import abc
import math

import torch

import mantrap.utility.shaping


###########################################################################
# Distributions ###########################################################
###########################################################################
class MultiModalDistribution(abc.ABC):

    def __init__(self, mus: torch.Tensor, log_pis: torch.Tensor, log_sigmas: torch.Tensor):
        self.mus = mus
        self.log_pis = log_pis
        self.log_sigmas = log_sigmas

    @abc.abstractmethod
    def log_prob(self, value):
        raise NotImplementedError


class GMM2D(MultiModalDistribution):
    """Gaussian Mixture Model using 2D Multivariate Gaussians each of as N components:
    Cholesky decompesition and affine transformation for sampling:

    .. math:: Z \\sim N(0, I)

    .. math:: S = \\mu + LZ

    .. math:: S \\sim N(\\mu, \\Sigma) \\rightarrow N(\\mu, LL^T)

    where :math:`L = chol(\\Sigma)` and

    .. math:: \\Sigma = \\left[ {\begin{array}{cc} \\sigma^2_x & \\rho \\sigma_x \\sigma_y \\
                        \\rho \\sigma_x \\sigma_y & \\sigma^2_y \\ \\end{array} } \\right]

    such that

    .. math:: L = chol(\\Sigma) = \\left[ {\\begin{array}{cc} \\sigma_x & 0 \\
                  \\rho \\sigma_y & \\sigma_y \\sqrt{1-\rho^2} \\ \\end{array} } \\right]

    Re-Implementation of GMM2D model used in GenTrajectron (B. Ivanovic, M. Pavone).

    :param log_pis: Log Mixing Proportions :math:`log(\\pi)`. [..., N]
    :param mus: Mixture Components mean :math:`\\mu`. [..., N * 2]
    :param log_sigmas: Log Standard Deviations :math:`log(\\sigma_d)`. [..., N * 2]
    :param corrs: Cholesky factor of correlation :math:`\\rho`. [..., N]
    """
    def __init__(self, mus: torch.Tensor, log_pis: torch.Tensor, log_sigmas: torch.Tensor, corrs: torch.Tensor):
        super(GMM2D, self).__init__(mus=mus, log_pis=log_pis, log_sigmas=log_sigmas)
        self.sigmas = torch.exp(self.log_sigmas)
        self.corrs = corrs
        self.one_minus_rho2 = torch.ones(1) - torch.pow(corrs, 2)

    def log_prob(self, values: torch.Tensor) -> torch.Tensor:
        """Calculates the log probability of a value using the PDF for bi-variate normal distributions:

        .. math::
            f(x | \\mu, \\sigma, \\rho)={\\frac {1}{2\\pi \\sigma _{x}\\sigma _{y}{\\sqrt {1-\\rho ^{2}}}}}\\exp
            \\left(-{\\frac {1}{2(1-\\rho ^{2})}}\\left[{\\frac {(x-\\mu _{x})^{2}}{\\sigma _{x}^{2}}}+
            {\\frac {(y-\\mu _{y})^{2}}{\\sigma _{y}^{2}}}-{\\frac {2\\rho (x-\\mu _{x})(y-\\mu _{y})}
            {\\sigma _{x}\\sigma _{y}}}\\right]\\right)

        :param values: The log probability density function is evaluated at those values.
        :returns: log probability of these values
        """
        value_un_squeezed = torch.unsqueeze(values, dim=-2)
        dx = value_un_squeezed - self.mus

        exp_nominator = ((torch.sum((dx / self.sigmas) ** 2, dim=-1)  # first and second term of exp nominator
                          - 2 * self.corrs * torch.prod(dx, dim=-1) / torch.prod(self.sigmas, dim=-1)))

        component_log_p = -(2 * math.log(2 * math.pi)
                            + torch.log(self.one_minus_rho2)
                            + 2 * torch.sum(self.log_sigmas, dim=-1)
                            + exp_nominator / self.one_minus_rho2) / 2

        return torch.logsumexp(self.log_pis + component_log_p, dim=-1)


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


def rotation_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Create rotation matrix from angle of vector from x -> y."""
    assert x.numel() == y.numel() == 2

    xy = y - x
    theta = torch.atan2(xy[1], xy[0])
    return torch.tensor([[torch.cos(theta), torch.sin(theta)],
                         [-torch.sin(theta), torch.cos(theta)]])


###########################################################################
# Logic ###################################################################
###########################################################################
def tensors_close(x: torch.Tensor, y: torch.Tensor, a_tol: float = 0.5) -> bool:
    return x.shape == y.shape and torch.all(torch.isclose(x, y, atol=a_tol))
