import abc
import math

import torch
import torch.distributions

import mantrap.utility.shaping


###########################################################################
# Distributions ###########################################################
###########################################################################
class VGMM2D(torch.distributions.Distribution):
    """Velocity-based Gaussian Mixture Model.

    Gaussian Mixture Model using 2D Multivariate Gaussians each of as N components:
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

    :param log_pis: Log Mixing Proportions (t_horizon, num_modes).
    :param mus: Mixture Components mean (t_horizon, num_modes, 2)
    :param log_sigmas: Log Standard Deviations (t_horizon, num_modes, 2)
    :param corrs: Cholesky factor of correlation (t_horizon, num_modes).
    """
    def __init__(self, mus: torch.Tensor, log_pis: torch.Tensor, log_sigmas: torch.Tensor, corrs: torch.Tensor):
        super(VGMM2D, self).__init__()
        t_horizon, self.components = log_pis.shape
        self.dimensions = 2
        assert mus.shape == (t_horizon, self.components, 2)
        assert log_sigmas.shape == (t_horizon, self.components, 2)
        assert corrs.shape == (t_horizon, self.components)

        # Distribution parameters.
        self.log_pis = log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True)  # [..., N]
        self.mus = mus
        self.log_sigmas = log_sigmas
        self.sigmas = torch.exp(self.log_sigmas)
        self.corrs = corrs

        # Pre-computations required for efficient sampling.
        self.one_minus_rho2 = torch.ones(1) - torch.pow(corrs, 2)
        self.pis_cat_dist = torch.distributions.Categorical(logits=log_pis)
        L1 = torch.stack([self.sigmas[..., 0], torch.zeros_like(log_pis)], dim=-1)
        L2 = torch.stack([self.sigmas[..., 1] * corrs, self.sigmas[..., 1] * self.one_minus_rho2], dim=-1)
        self.L = torch.stack([L1, L2], dim=-2)

    def log_prob(self, velocities: torch.Tensor) -> torch.Tensor:
        """Calculates the log probability of a value using the PDF for bi-variate normal distributions:

        .. math::
            f(x | \\mu, \\sigma, \\rho)={\\frac {1}{2\\pi \\sigma _{x}\\sigma _{y}{\\sqrt {1-\\rho ^{2}}}}}\\exp
            \\left(-{\\frac {1}{2(1-\\rho ^{2})}}\\left[{\\frac {(x-\\mu _{x})^{2}}{\\sigma _{x}^{2}}}+
            {\\frac {(y-\\mu _{y})^{2}}{\\sigma _{y}^{2}}}-{\\frac {2\\rho (x-\\mu _{x})(y-\\mu _{y})}
            {\\sigma _{x}\\sigma _{y}}}\\right]\\right)

        :param velocities: The log probability density function is evaluated at those velocities.
        :returns: log probability of these values
        """
        dx = velocities - self.mus
        exp_nominator = ((torch.sum((dx / self.sigmas) ** 2, dim=-1)  # first and second term of exp nominator
                          - 2 * self.corrs * torch.prod(dx, dim=-1) / torch.prod(self.sigmas, dim=-1)))

        component_log_p = -(2 * math.log(2 * math.pi)
                            + torch.log(self.one_minus_rho2)
                            + 2 * torch.sum(self.log_sigmas, dim=-1)
                            + exp_nominator / self.one_minus_rho2) / 2

        return torch.logsumexp(self.log_pis + component_log_p, dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        """Generates a sample_shape shaped re-parameterized sample or sample_shape shaped batch of
        re-parameterized  samples if the distribution parameters are batched.

        :param sample_shape: Shape of the samples
        :return: Samples from the GMM in velocity space.
        """
        samples = torch.randn(size=sample_shape + self.mus.shape).unsqueeze(dim=-1)
        mvn_samples = self.mus + torch.matmul(self.L, samples).squeeze(dim=-1)
        component_cat_samples = self.pis_cat_dist.sample(sample_shape)
        selector = torch.eye(self.components)[component_cat_samples].unsqueeze(dim=-1)
        return torch.sum(mvn_samples * selector, dim=-2, keepdim=True)

    @property
    def mean(self):
        return self.mus


###########################################################################
# Numerical Methods #######################################################
###########################################################################
def derivative_numerical(x: torch.Tensor, dt: float) -> torch.Tensor:
    """Compute the derivative numerically by applying numerical differentiation.

    ... math:: \\frac{d}{dt} x = \\frac{x_t - x_{t-1}}{dt}

    :param x: tensor to differentiate numerically (num_ados, num_samples, T, num_modes, 2) or (T, 2).
    :param dt: differentiation time-step.
    :returns: differentiated tensor x (num_ados, num_samples, T - 1, num_modes, 2) or (T-1, 2).
    """
    if len(x.shape) == 2:
        return (x[1:, :] - x[:-1, :]) / dt
    else:
        return (x[..., 1:, :, :] - x[..., :-1, :, :]) / dt


def integrate_numerical(x: torch.Tensor, dt: float, x0: torch.Tensor) -> torch.Tensor:
    """Compute the integral of a vector numerically by applying Euler forward integration.

    ... math:: \\int x dt = x0 + \\sum_t x_t * dt

    :param x: vector to differentiate numerically (num_ados, num_samples, T - 1, num_modes, 2) or (T-1, 2).
    :param dt: differentiation time-step.
    :param x0: initial condition (num_ados, 2).
    :returns: differentiated vector x (num_ados, num_samples, T, num_modes, 2) or (T, 2).
    """
    x_shape = x.shape
    x_size = len(x_shape)


    if x_size == 2:
        padding = torch.zeros((1, 2))
        axis = 0

    else:
        assert x0.shape[0] == x_shape[0]
        if len(x0.shape) == 2:
            x0 = x0.view(-1, *tuple([1] * (x_size - 3)), 1, 2)
        if x_size == 4:
            padding = torch.zeros((x_shape[0], 1, x_shape[-2], 2))
        elif x_size == 5:
            padding = torch.zeros((x_shape[0], x_shape[1], 1, x_shape[-2], 2))
        else:
            raise NotImplementedError(f"Integral not implemented for x_size = {x_size} !")
        axis = x_size - 3  # 4 -> 1, 5 -> 2

    x_padded = torch.cat((padding, x), dim=axis)
    x_int = torch.cumsum(x_padded, dim=axis) * dt + x0
    return x_int


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


def rotation_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Create rotation matrix from angle of vector from x -> y."""
    assert x.numel() == y.numel() == 2

    xy = y - x
    theta = torch.atan2(xy[1], xy[0])
    return torch.tensor([[torch.cos(theta), torch.sin(theta)],
                         [-torch.sin(theta), torch.cos(theta)]])
