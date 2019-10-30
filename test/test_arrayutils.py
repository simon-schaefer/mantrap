from typing import Tuple

import numpy as np
import pytest

import murseco.utility.array


@pytest.mark.parametrize("shape", [(2, 2), (4, 2, 2), (1, 1)])
def test_rand_inv_pos_symmetric_matrix(shape: Tuple):
    a = murseco.utility.array.rand_inv_pos_symmetric_matrix(*shape)

    if len(shape) > 2:
        assert np.all([np.linalg.det(ai) != 0 for ai in a])
        assert np.all([np.array_equal(ai, ai.T) for ai in a])
    else:
        assert np.linalg.det(a) != 0
        assert np.array_equal(a.T, a)
    assert np.alltrue(a >= 0)


def test_spline_re_sampling():
    a = np.vstack((np.ones((11,)), np.linspace(0, 10, 11))).T
    a_re = murseco.utility.array.spline_re_sampling(a, inter_distance=0.1)

    assert np.all(np.isclose(np.sqrt(np.sum(np.diff(a_re, axis=0) ** 2, axis=1)), 0.1, atol=0.02))
    assert np.isclose(np.linalg.norm(a_re[0, :] - a[0, :]), 0, atol=0.1)
    assert np.isclose(np.linalg.norm(a_re[-1, :] - a[-1, :]), 0, atol=0.1)


def test_linear_sub_sampling():
    a = np.zeros((10, 5, 5))
    for i in range(a.shape[0]):
        a[i, :, :] = np.ones((5, 5)) * i
    a_sub = murseco.utility.array.linear_sub_sampling(a, num_sub_samples=101)

    for i in range(100):
        assert np.array_equal(a_sub[i, :, :], np.ones((5, 5)) * i * 0.1)
