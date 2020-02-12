import time

import numpy as np
import pytest
import torch

from mantrap.utility.maths import Derivative2, lagrange_interpolation
from mantrap.utility.primitives import straight_line_primitive
from mantrap.utility.utility import build_trajectory_from_positions


###########################################################################
# Utility Testing #########################################################
###########################################################################
def test_build_trajectory_from_positions():
    positions = straight_line_primitive(horizon=11, start_pos=torch.zeros(2), end_pos=torch.ones(2) * 10)
    trajectory = build_trajectory_from_positions(positions, dt=1.0, t_start=0.0)

    assert torch.all(torch.eq(trajectory[:, 0:2], positions))
    assert torch.all(torch.eq(trajectory[1:, 3:5], torch.ones(10, 2)))


###########################################################################
# Derivative2 Testing #####################################################
###########################################################################
@pytest.mark.parametrize("horizon", [7, 10])
def test_derivative_2(horizon: int):
    diff = Derivative2(horizon=horizon, dt=1.0)

    for k in range(1, horizon - 1):
        assert np.array_equal(diff._diff_mat[k, k - 1: k + 2], np.array([1, -2, 1]))

    # Constant velocity -> zero acceleration (first and last point are skipped (!)).
    x = straight_line_primitive(horizon, torch.zeros(2), torch.tensor([horizon - 1, 0])).detach().numpy()
    assert np.array_equal(diff.compute(x), np.zeros((horizon, 2)))

    t_step = int(horizon / 2)
    x = np.zeros((horizon, 2))
    x[t_step, 0] = 10
    x[t_step - 1, 0] = x[t_step + 1, 0] = 5
    x[t_step - 2, 0] = x[t_step + 2, 0] = 2
    a = diff.compute(x)

    assert np.all(a[: t_step - 3, 0] == 0) and np.all(a[t_step + 4:, 0] == 0)  # impact of acceleration
    assert np.all(a[:, 1] == 0)  # no acceleration in y-direction
    assert a[t_step, 0] < 0 and a[t_step + 1, 0] > 0 and a[t_step - 1, 0] > 0  # peaks


def test_derivative_2_conserve_shape():
    x = np.zeros((1, 1, 10, 6))
    diff = Derivative2(horizon=10, dt=1.0, num_axes=4)
    x_ddt = diff.compute(x)

    assert x.shape == x_ddt.shape


def test_derivative_2_runtime():
    x = np.random.rand(5, 1, 10, 6)
    diff = Derivative2(horizon=10, dt=1.0, num_axes=4)

    start_time = time.time()
    diff.compute(x)
    run_time = time.time() - start_time

    assert run_time < 1e-3  # > 1000 Hz


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

    start = torch.tensor([0, 0]).float()
    mid = torch.tensor([5.0, 5.0], requires_grad=True).float()
    end = torch.tensor([10, 0]).float()
    points = torch.stack((start, mid, end)).float()

    points_up = lagrange_interpolation(control_points=points, num_samples=100)

    # Test trajectory shape and it itself.
    assert len(points_up.shape) == 2 and points_up.shape[1] == 2 and points_up.shape[0] == 100
    for n in range(100):
        assert start[0] <= points_up[n, 0] <= end[0]
        assert min(start[1], mid[1], end[1]) <= points_up[n, 1] <= max(start[1], mid[1], end[1])

    # Test derivatives of up-sampled trajectory.
    for n in range(1, 100):  # first point has no gradient since (0, 0) control point
        dx = torch.autograd.grad(points_up[n, 0], mid, retain_graph=True)[0]
        assert torch.all(torch.eq(dx, torch.zeros(2)))  # created by linspace operation
        dy = torch.autograd.grad(points_up[n, 1], mid, retain_graph=True)[0]
        assert not torch.all(torch.eq(dy, torch.zeros(2)))  # created by interpolation
