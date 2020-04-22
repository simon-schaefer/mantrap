import pytest

from mantrap.utility.maths import *
from mantrap.utility.primitives import square_primitives, straight_line


###########################################################################
# Straight Line Testing ###################################################
###########################################################################
def test_build_trajectory_from_positions():
    path = straight_line(start_pos=torch.zeros(2), end_pos=torch.ones(2) * 10, steps=11)
    velocities = torch.cat(((path[1:, 0:2] - path[:-1, 0:2]) / 1.0, torch.zeros((1, 2))))
    trajectory = torch.cat((path, velocities), dim=1)

    assert torch.all(torch.eq(trajectory[:, 0:2], path))
    assert torch.all(torch.eq(trajectory[:-1, 2:4], torch.ones(10, 2)))


###########################################################################
# Derivative2 Testing #####################################################
###########################################################################
@pytest.mark.parametrize("horizon", [7, 10])
def test_derivative_2(horizon: int):
    diff = Derivative2(horizon=horizon, dt=1.0)

    for k in range(1, horizon - 1):
        assert torch.all(torch.eq(diff._diff_mat[k, k - 1: k + 2], torch.tensor([1, -2, 1]).float()))

    # Constant velocity -> zero acceleration (first and last point are skipped (!)).
    x = straight_line(torch.zeros(2), torch.tensor([horizon - 1, 0]), horizon)
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
    diff = Derivative2(horizon=10, dt=1.0, num_axes=4)
    x_ddt = diff.compute(x)

    assert x.shape == x_ddt.shape


def test_derivative_2_velocity():
    x = torch.rand((2, 5))
    diff = Derivative2(horizon=2, dt=1.0, velocity=True)
    x_ddt = diff.compute(x[:, 2:4])

    assert torch.all(torch.isclose(x_ddt[0, :], torch.zeros(2)))
    assert torch.all(torch.isclose(x_ddt[1, :], (x[1, 2:4] - x[0, 2:4]) / diff._dt))


###########################################################################
# Interpolation Testing ###################################################
###########################################################################
def test_lagrange_interpolation():
    """Lagrange interpolation using Vandermonde Approach. Vandermonde finds the Lagrange parameters by solving
    a matrix equation X a = Y with known control point matrices (X,Y) and parameter vector a, therefore is fully
    differentiable. Also Lagrange interpolation guarantees to pass every control point, but performs poorly in
    extrapolation (which is however not required for trajectory fitting, since the trajectory starts and ends at
    defined control points.

    Source: http://www.maths.lth.se/na/courses/FMN050/media/material/lec8_9.pdf"""

    start = torch.tensor([0.0, 0.0])
    mid = torch.tensor([5.0, 5.0], requires_grad=True)
    end = torch.tensor([10.0, 0.0])
    points = torch.stack((start, mid, end))

    points_up = lagrange_interpolation(control_points=points, num_samples=100, deg=3)

    # Test trajectory shape and it itself.
    assert len(points_up.shape) == 2 and points_up.shape[1] == 2 and points_up.shape[0] == 100
    for n in range(100):
        assert start[0] <= points_up[n, 0] <= end[0]
        assert min(start[1], mid[1], end[1]) <= points_up[n, 1] + 0.0001
        assert points_up[n, 1] - 0.0001 <= max(start[1], mid[1], end[1])

    # Test derivatives of up-sampled trajectory.
    for n in range(1, 100):  # first point has no gradient since (0, 0) control point
        dx = torch.autograd.grad(points_up[n, 0], mid, retain_graph=True)[0]
        assert torch.all(torch.eq(dx, torch.zeros(2)))  # created by lin-space operation
        dy = torch.autograd.grad(points_up[n, 1], mid, retain_graph=True)[0]
        assert not torch.all(torch.eq(dy, torch.zeros(2)))  # created by interpolation


@pytest.mark.xfail(raises=RuntimeError)
def test_lagrange_singularity():
    start = torch.tensor([0.0, 0.0])
    mid = torch.tensor([0.0, 5.0], requires_grad=True)
    end = torch.tensor([10.0, 0.0])
    points = torch.stack((start, mid, end))  # singular matrix (!)

    points_up = lagrange_interpolation(control_points=points, num_samples=10)
    for n in range(1, 10):
        torch.autograd.grad(points_up[n, 0], mid, retain_graph=True)
        torch.autograd.grad(points_up[n, 1], mid, retain_graph=True)


###########################################################################
# Primitives ##############################################################
###########################################################################
@pytest.mark.parametrize("num_points", [5, 10])
def test_square_primitives(num_points: int):
    position, velocity, goal = torch.tensor([-5.0, 0.0]), torch.tensor([1.0, 0.0]), torch.tensor([20.0, 0.0])
    primitives = square_primitives(start=position, end=goal, dt=1.0, steps=num_points)

    assert primitives.shape[1] == num_points
    # for m in range(primitives.shape[0]):
    #     for i in range(1, num_points - 1):
    #         distance = torch.norm(primitives[m, i, :] - primitives[m, i - 1, :])
    #         distance_next = torch.norm(primitives[m, i + 1, :] - primitives[m, i, :])
    #         if torch.isclose(distance_next, torch.zeros(1), atol=0.1):
    #             continue
    #         tolerance = agent_speed_max / 10
    #         assert torch.isclose(distance, torch.tensor([agent_speed_max]).double(), atol=tolerance)  # dt = 1.0

    # The center primitive should be a straight line, therefore the one with largest x-expansion, since we are moving
    # straight in x-direction. Similarly the first primitive should have the largest expansion in y direction, the
    # last one the smallest.
    assert all([primitives[1, -1, 0] >= primitives[i, -1, 0] for i in range(primitives.shape[0])])
    assert all([primitives[0, -1, 1] >= primitives[i, -1, 1] for i in range(primitives.shape[0])])
    assert all([primitives[-1, -1, 1] <= primitives[i, -1, 1] for i in range(primitives.shape[0])])


###########################################################################
# Shapes Testing ##########################################################
###########################################################################
def test_circle_sampling():
    circle = Circle(center=torch.rand(2), radius=3.0)
    samples = circle.samples(num_samples=100)

    assert samples.shape == (100, 2)

    min_x, min_y = min(samples[:, 0]), min(samples[:, 1])
    max_x, max_y = max(samples[:, 0]), max(samples[:, 1])
    radius_numeric = (max_x - min_x) / 2

    assert torch.isclose((max_y - min_y) / 2, radius_numeric, atol=0.1)  # is circle ?
    assert torch.isclose(torch.tensor(circle.radius), radius_numeric, atol=0.1)  # same circle ?


def test_circle_intersection():
    circle = Circle(center=torch.tensor([4, 5]), radius=2.0)

    # These circles do not intersect, since the distance between the centers is larger than 1 + 2 = 3.
    circle_not = Circle(center=torch.tensor([0, 5]), radius=1.0)
    assert not circle.does_intersect(circle_not)

    # These circles do intersect, since the distance between the centers is smaller than 3 + 2 = 5.
    circle_is = Circle(center=torch.tensor([2, 3]), radius=3.0)
    assert circle.does_intersect(circle_is)
