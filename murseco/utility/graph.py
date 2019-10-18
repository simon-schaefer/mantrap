from typing import Tuple, Union

import numpy as np


def grid_cont_2_discrete(
    coordinate: np.ndarray, meshgrid: Tuple[np.ndarray, np.ndarray]
) -> Tuple[Union[int, None], Union[int, None]]:
    """Convert a continuous coordinate to a grid coordinate in the given meshgrid by exploiting the monotonicity,
    linearity and repetitiveness of both of the meshgrid arrays.

    :argument coordinate: continuous coordinate to transform in 2D (2,).
    :argument meshgrid: numpy meshgrid (x, y).
    :returns coordinate in discrete meshgrid coordinates
    """
    assert coordinate.size == 2, "coordinate must be two-dimensional"

    x_min, x_max, x_size = meshgrid[0][0, 0], meshgrid[0][0, -1], meshgrid[0].shape[1]
    y_min, y_max, y_size = meshgrid[1][0, 0], meshgrid[1][-1, 0], meshgrid[1].shape[0]
    if (not x_min <= coordinate[0] <= x_max) or (not y_min <= coordinate[1] <= y_max):
        return None, None

    x = (coordinate[0] - x_min) / (x_max - x_min) * (x_size - 1)  # start index at 0
    y = (coordinate[1] - y_min) / (y_max - y_min) * (y_size - 1)  # start index at 0
    return int(round(x)), int(round(y))
