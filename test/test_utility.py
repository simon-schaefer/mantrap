import pytest
import torch

import mantrap.utility.maths


###########################################################################
# Derivative2 Testing #####################################################
###########################################################################
def test_derivative_conserve_shape():
    x = torch.rand((1, 10, 1, 2))
    x_ddt = mantrap.utility.maths.derivative_numerical(x, dt=1.0)
    assert x_ddt.shape == torch.Size([1, 9, 1, 2])


def test_integral_conserve_shape():
    x = torch.rand((1, 10, 1, 2))
    x_int = mantrap.utility.maths.integrate_numerical(x, dt=1.0, x0=torch.rand((1, 2)))
    assert x_int.shape == torch.Size([1, 11, 1, 2])


def test_derivative():
    x = torch.rand((1, 2, 1, 5))
    x_ddt = mantrap.utility.maths.derivative_numerical(x[:, :, :, 2:4], dt=1.0)
    assert torch.all(torch.isclose(x_ddt[:, 0, :, :], (x[:, 1, :, 2:4] - x[:, 0, :, 2:4]) / 1.0))

    x = torch.rand((2, 5))
    x_ddt = mantrap.utility.maths.derivative_numerical(x[:, 2:4], dt=1.0)
    assert torch.all(torch.isclose(x_ddt[0, :], (x[1, 2:4] - x[0, 2:4]) / 1.0))


###########################################################################
# Shapes Testing ##########################################################
###########################################################################
def test_circle_sampling():
    circle = mantrap.utility.maths.Circle(center=torch.rand(2), radius=3.0)
    samples = circle.samples(num_samples=100)

    assert samples.shape == (100, 2)

    min_x, min_y = min(samples[:, 0]), min(samples[:, 1])
    max_x, max_y = max(samples[:, 0]), max(samples[:, 1])
    radius_numeric = (max_x - min_x) / 2

    assert torch.isclose((max_y - min_y) / 2, radius_numeric, atol=0.1)  # is circle ?
    assert torch.isclose(torch.tensor(circle.radius), radius_numeric, atol=0.1)  # same circle ?


def test_circle_intersection():
    circle = mantrap.utility.maths.Circle(center=torch.tensor([4, 5]), radius=2.0)

    # These circles do not intersect, since the distance between the centers is larger than 1 + 2 = 3.
    circle_not = mantrap.utility.maths.Circle(center=torch.tensor([0, 5]), radius=1.0)
    assert not circle.does_intersect(circle_not)

    # These circles do intersect, since the distance between the centers is smaller than 3 + 2 = 5.
    circle_is = mantrap.utility.maths.Circle(center=torch.tensor([2, 3]), radius=3.0)
    assert circle.does_intersect(circle_is)
