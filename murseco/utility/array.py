import numpy as np
import scipy.interpolate


def rand_inv_pos_symmetric_matrix(*size) -> np.ndarray:
    """Create a random invertible, positive definite and  symmetric matrix with the given dimensions, by breaking it
    down to 2D (last two dimensions) and repeating this matrix over the other dimensions.

    :argument size: sequence of integers stating matrix shape e.g. *size = 1, 3, 4.
    """
    assert len(size) > 1, "size must be at least two-dimensional"
    assert size[-1] == size[-2], "last two dimensions must be identical in order to be quadratic"

    a = np.random.rand(size[-2], size[-1])
    a = a.T * a + np.eye(size[-1]) * np.amax(a)
    if len(size) == 2:
        return a
    else:
        return np.tile(a, (*size[:-2], 1, 1))


def spline_re_sampling(array: np.ndarray, num_sub_samples: int = 2000, inter_distance: float = None) -> np.ndarray:
    """Use spline-based interpolation to higher resolute an array, e.g. an array of two-dimensional path points.
    Therefore first interpolate the array as spline (cubic) and re-sample it using much more samples as before. Then
    re-sample so that each point in the array is equal-distant to its neighbours.

    :argument array: array to sub- and re-sample (num_points, num_dimensions).
    :argument inter_distance: distance between each point and its neighbours in output array (no re-sampling if None).
    :argument num_sub_samples: number of re-sampling points (increasing accuracy in re-sampling).
    :returns interpolated and re-sampled equal-distant array (..., num_dimensions).
    """
    # Cubic interpolation and re-sampling array to sub-sampled points. .
    psi = np.linspace(0, 1, num=array.shape[0])
    interpolator = scipy.interpolate.interp1d(psi, array, kind="cubic", axis=0)
    points = interpolator(np.linspace(0, 1, num=num_sub_samples))

    # Re-sample path so that each point is equal-distant from its neighbours, with the distance specified in the
    # functions arguments.
    if inter_distance is not None:
        inter_point_distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
        inter_point_distances = np.asarray(np.insert(inter_point_distances, 0, 0) / inter_distance, dtype=int)
        _, inter_point_index = np.unique(inter_point_distances, return_index=True)
        points = points[inter_point_index, :]

    return points


def linear_sub_sampling(array: np.ndarray, num_sub_samples: int = 2000) -> np.ndarray:
    """Linear interpolation between a sequence of images, i.e. I_interpolated = I_prev + (I_next - I_prev) * delta.
    In order to create num_sub_sampling interpolated images create a numpy lin-space from 0 to the current number of
    images in the image sequence, divided into sub-steps by the number of sub-samples.

    :argument array: array to sub-sample (num_samples, x_dim, y_dim).
    :argument num_sub_samples: number of returned interpolated samples.
    :returns interpolated array (num_sub_samples, x_dim, y_dim).
    """
    assert len(array.shape) == 3, "merely arrays in shape (N, X, Y) are supported"
    assert num_sub_samples >= array.shape[0], "number of sub-samples must be larger than current number of samples"

    num_samples = array.shape[0]
    sub_samples = np.zeros((num_sub_samples, array.shape[1], array.shape[2]))
    for i, n in enumerate(np.linspace(0, num_samples, num=num_sub_samples)):
        n_down, n_up = min(max(int(n), 0), num_samples - 2), min(max(int(n) + 1, 1), num_samples - 1)
        print(n_down, n_up)
        sub_samples[i, :, :] = array[n_down, :, :] + (array[n_up, :, :] - array[n_down, :, :]) * (n - n_down)
    return sub_samples
