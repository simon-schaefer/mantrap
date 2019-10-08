import numpy as np


def filter_by_range(array: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Filters out columns that do not fulfill lower_i < x_i < upper_i for any row (i = [0, N]) in the array.

     :argument array: Array to filter (NxM).
     :argument lower: Lower bound for array values for each dimension (Mx1).
     :argument upper: Upper bound for array values for each dimension (Mx1).
     :return filtered: filtered array (LxN) with L <= M.
     """
    ncols = array.shape[1]
    assert lower.size == ncols and upper.size == ncols, "bounds have invalid dimension"
    mask = (np.subtract(array, upper) <= 0) * (np.subtract(array, lower) >= 0)
    filtered = array[np.sum(mask, axis=1) == ncols, :]
    return np.reshape(filtered, (filtered.size // ncols, ncols))


def grid_points_from_ranges(xrange: np.ndarray, yrange: np.ndarray) -> np.ndarray:
    """Build grid points from range in x and y direction i.e. for an array containing every combination
    from x and y points (MxN).

    :argument xrange: 1d array containing x coordinates (Mx1).
    :argument yrange: 1d array containing y coordinates (Nx1).
    :return grid_points: grid points array (MxN).
    """
    return np.transpose([np.tile(xrange, len(yrange)), np.repeat(yrange, len(xrange))])


def rand_invsymmpos(*size) -> np.ndarray:
    """Create a random invertible, positive definite and  symmetric matrix with the given dimensions, by breaking it
    down to 2D (last two dimensions) and repeating this matrix over the other dimensions.

    :argument size: sequence of integers stating matrix shape e.g. *size = 1, 3, 4.
    """
    assert len(size) > 2, "size must be at least three-dimensional"
    a = np.random.rand(size[-2], size[-1])
    a = a.T * a + np.eye(size[-1]) * np.amax(a)
    return np.tile(a, (*size[:-2], 1, 1))

