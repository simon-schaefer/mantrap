import numpy as np
import pytest

import murseco.utility.array


@pytest.mark.parametrize(
    "raw, lower, upper, target",
    [
        (np.array([[2, 3, 4, 5, 6]]).T, np.array([1]), np.array([5]), np.array([[2, 3, 4, 5]]).T),
        (
            np.array([[2, 3, 4, 5, 6], [2, 3, 4, 5, 6]]).T,
            np.array([1, 3]),
            np.array([5, 5]),
            np.array([[3, 4, 5], [3, 4, 5]]).T,
        ),
    ],
)
def test_arrayops_filter_by_range(raw: np.ndarray, lower: np.ndarray, upper: np.ndarray, target: np.ndarray):
    filtered = murseco.utility.array.filter_by_range(raw, lower, upper)
    print(np.shape(filtered))
    assert np.array_equal(filtered, target)


def test_arrayops_grid_points_from_range():
    xrange, yrange = np.linspace(0, 5, 6), np.linspace(0, 2, 3)
    grid_points = murseco.utility.array.grid_points_from_ranges(xrange, yrange).tolist()
    grid_points_list = [[x0, y0] for x0 in xrange for y0 in yrange]
    assert all([xy in grid_points for xy in grid_points_list])
