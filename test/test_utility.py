import pytest
import torch

import mantrap.utility.maths


###########################################################################
# Straight Line Testing ###################################################
###########################################################################
def test_build_trajectory_from_positions():
    path = mantrap.utility.maths.straight_line(start=torch.zeros(2), end=torch.ones(2) * 10, steps=11)
    velocities = torch.cat(((path[1:, 0:2] - path[:-1, 0:2]) / 1.0, torch.zeros((1, 2))))
    trajectory = torch.cat((path, velocities), dim=1)

    assert torch.all(torch.eq(trajectory[:, 0:2], path))
    assert torch.all(torch.eq(trajectory[:-1, 2:4], torch.ones(10, 2)))


###########################################################################
# Derivative2 Testing #####################################################
###########################################################################
@pytest.mark.parametrize("horizon", [7, 10])
def test_derivative_2(horizon: int):
    diff = mantrap.utility.maths.Derivative2(horizon=horizon, dt=1.0)

    for k in range(1, horizon - 1):
        assert torch.all(torch.eq(diff._diff_mat[k, k - 1: k + 2], torch.tensor([1, -2, 1]).float()))

    # Constant velocity -> zero acceleration (first and last point are skipped (!)).
    x = mantrap.utility.maths.straight_line(torch.zeros(2), torch.tensor([horizon - 1, 0]), horizon)
    assert torch.all(torch.eq(diff.compute(x), torch.zeros((horizon, 2))))

    t_step = int(horizon / 2)
    x = torch.zeros((horizon, 2))
    x[t_step, 0] = 10
    x[t_step - 1, 0] = x[t_step + 1, 0] = 5
    x[t_step - 2, 0] = x[t_step + 2, 0] = 2
    a = diff.compute(x)

    assert torch.all(a[: t_step - 3, 0] == 0) and torch.all(a[t_step + 4:, 0] == 0)  # impact of acceleration
    assert torch.all(a[:, 1] == 0)  # no acceleration in y-direction
    assert a[t_step, 0] < 0 and a[t_step + 1, 0] > 0 and a[t_step - 1, 0] > 0  # peaks


def test_derivative_2_conserve_shape():
    x = torch.zeros((1, 1, 10, 6))
    diff = mantrap.utility.maths.Derivative2(horizon=10, dt=1.0, num_axes=4)
    x_ddt = diff.compute(x)

    assert x.shape == x_ddt.shape


def test_derivative_2_velocity():
    x = torch.rand((2, 5))
    diff = mantrap.utility.maths.Derivative2(horizon=2, dt=1.0, velocity=True)
    x_ddt = diff.compute(x[:, 2:4])

    assert torch.all(torch.isclose(x_ddt[0, :], torch.zeros(2)))
    assert torch.all(torch.isclose(x_ddt[1, :], (x[1, 2:4] - x[0, 2:4]) / diff._dt))


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
