from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest

import murseco.utility.arrayops
import murseco.utility.io
import murseco.utility.stats
import murseco.utility.visualization


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
    filtered = murseco.utility.arrayops.filter_by_range(raw, lower, upper)
    print(np.shape(filtered))
    assert np.array_equal(filtered, target)


def test_arrayops_grid_points_from_range():
    xrange, yrange = np.linspace(0, 5, 6), np.linspace(0, 2, 3)
    grid_points = murseco.utility.arrayops.grid_points_from_ranges(xrange, yrange).tolist()
    grid_points_list = [[x0, y0] for x0 in xrange for y0 in yrange]
    assert all([xy in grid_points for xy in grid_points_list])


@pytest.mark.parametrize(
    "mu, sigma, point, target",
    [
        (np.array([1.0, 0.0]), np.eye(2), (0, 0), 0.096),
        (np.array([0.0, 0.0]), np.eye(2) * 0.01, (10, 10), 0.0),
        (np.array([0.0, 0.0]), np.diag([0.01, 10.0]), (0, 10), 0.0033),
    ],
)
def test_stats_gaussian2d_pdf_at(mu: np.ndarray, sigma: np.ndarray, point: Tuple[float, float], target: float):
    distribution = murseco.utility.stats.Gaussian2D(mu, sigma)
    assert np.isclose(distribution.pdf_at(point[0], point[1]), target, rtol=0.1)


def test_stats_gaussian2d_json():
    gauss_1 = murseco.utility.stats.Gaussian2D(np.array([4.1, -1.39]), np.eye(2) * 0.14)
    cache_path = murseco.utility.io.path_from_home_directory("test/cache/gaussian2d_test.json")
    gauss_1.to_json(cache_path)
    gauss_2 = murseco.utility.stats.Gaussian2D.from_json(cache_path)
    assert gauss_1.description == gauss_2.description
    assert gauss_1.pdf_at(0.0, 0.0) == gauss_2.pdf_at(0.0, 0.0)


@pytest.mark.parametrize(
    "mus, sigmas, weights, point, target",
    [
        (np.array([[1, 0], [0, 0]]), np.stack((np.eye(2), np.eye(2))), np.array([1, 0]), (0, 0), 0.096),
        (np.array([[0, 0], [0, 0]]), np.stack((np.eye(2) * 0.01, np.eye(2))), np.array([1, 0]), (10, 10), 0.0),
        (np.array([[0, 0], [0, 0]]), np.stack((np.diag([0.01, 10]), np.eye(2))), np.array([1, 0]), (0, 10), 0.0033),
    ],
)
def test_stats_gmm2d_pdf_at(
    mus: np.ndarray, sigmas: np.ndarray, weights: np.ndarray, point: Tuple[float, float], target: float
):
    distribution = murseco.utility.stats.GMM2D(mus, sigmas, weights)
    assert np.isclose(distribution.pdf_at(point[0], point[1]), target, rtol=0.1)


def test_stats_gmm2d_json():
    mus = np.array([[4.1, -1.39], [2.1, -9.1]])
    sigmas = np.stack((np.eye(2) * 0.14, np.eye(2)))
    weights = np.array([0.1, 1.0])
    gmm_1 = murseco.utility.stats.GMM2D(mus, sigmas, weights)
    cache_path = murseco.utility.io.path_from_home_directory("test/cache/gmm2d_test.json")
    gmm_1.to_json(cache_path)
    gmm_2 = murseco.utility.stats.GMM2D.from_json(cache_path)
    assert gmm_1.description == gmm_2.description
    assert gmm_1.pdf_at(0.0, 0.0) == gmm_2.pdf_at(0.0, 0.0)


@pytest.mark.parametrize(
    "mus, sigmas, weights, index",
    [
        (np.array([[4, 2], [0, 0]]), np.stack((np.eye(2), np.eye(2))), np.array([1, 1]), "1"),
        (np.array([[5, 5], [-5, -5]]), np.stack((np.eye(2) * 0.01, np.eye(2) * 0.1)), np.array([1, 1]), "2"),
        (np.array([[0, 0], [0, 0]]), np.stack((np.diag([0.01, 10]), np.eye(2))), np.array([1, 0]), "3"),
    ],
)
def test_visualization_plot_pdf2d(mus: np.ndarray, sigmas: np.ndarray, weights: np.ndarray, index: str):
    cache_path = murseco.utility.io.path_from_home_directory(f"test/cache/gmm2d_test_{index}.png")
    distribution = murseco.utility.stats.GMM2D(mus, sigmas, weights)
    fig, ax = plt.subplots()
    murseco.utility.visualization.plot_pdf2d(fig, ax, distribution, (-10, 10))
    plt.savefig(cache_path)
