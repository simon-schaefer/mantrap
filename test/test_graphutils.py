from typing import Tuple

import numpy as np
import pytest

from murseco.utility.io import path_from_home_directory
import murseco.utility.graph


@pytest.mark.parametrize(
    "coordinate, discrete_expected",
    [(np.zeros(2), (0, 0)), (np.array([5.32, 4.1]), (5, 4)), (np.array([10, 9]), (10, 9))],
)
def test_graphutils_cont_to_discrete(coordinate: np.ndarray, discrete_expected: Tuple[int, int]):
    x_grid, y_grid = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
    discrete = murseco.utility.graph.grid_cont_2_discrete(coordinate, (x_grid, y_grid))
    assert len(discrete) == 2
    assert discrete[0] == discrete_expected[0]
    assert discrete[1] == discrete_expected[1]
    assert np.isclose(x_grid[discrete[1], discrete[0]], coordinate[0], atol=0.5)
    assert np.isclose(y_grid[discrete[1], discrete[0]], coordinate[1], atol=0.5)
