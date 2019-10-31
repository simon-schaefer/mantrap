from typing import Tuple

import numpy as np
import pytest
import scipy.cluster.vq

import murseco.utility.array
import murseco.utility.io
import murseco.utility.stats


@pytest.mark.parametrize(
    "position, point, target",
    [
        (np.array([1.0, 0.0]), (0.0, 0.0), 0),
        (np.array([1.0, 0.0]), (0.999999, 0.0), 100),
        (np.array([1.0, 0.0]), (1.0, 0.0), 100),
    ],
)  # np.inf = 100 (comp. definition of Point2D distribution)
def test_point2d_pdf_at(position: np.ndarray, point: Tuple[float, float], target: float):
    distribution = murseco.utility.stats.Point2D(position)
    assert np.isclose(distribution.pdf_at(point[0], point[1]), target, rtol=0.1)


def test_point2d_sample():
    xy = np.array([4.1, 1.23])
    distribution = murseco.utility.stats.Point2D(xy)
    assert all([np.array_equal(s, xy) for s in distribution.sample(2)])


@pytest.mark.parametrize(
    "mu, sigma, point, target",
    [
        (np.array([1.0, 0.0]), np.eye(2), (0, 0), 0.096),
        (np.array([0.0, 0.0]), np.eye(2) * 0.01, (10, 10), 0.0),
        (np.array([0.0, 0.0]), np.diag([0.01, 10.0]), (0, 10), 0.0033),
    ],
)
def test_gaussian2d_pdf_at(mu: np.ndarray, sigma: np.ndarray, point: Tuple[float, float], target: float):
    distribution = murseco.utility.stats.Gaussian2D(mu, sigma)
    assert np.isclose(distribution.pdf_at(point[0], point[1]), target, rtol=0.1)


def test_point2d_json():
    point_1 = murseco.utility.stats.Point2D(np.array([4.1, -1.39]))
    cache_path = murseco.utility.io.path_from_home_directory("test/cache/point2d_test.json")
    point_1.to_json(cache_path)
    point_2 = murseco.utility.stats.Point2D.from_json(cache_path)
    assert point_1.summary() == point_2.summary()


def test_gaussian2d_sample():
    np.random.seed(0)
    mu, sigma = np.array([10, 0]), np.diag([0.01, 2])
    distribution = murseco.utility.stats.Gaussian2D(mu, sigma)
    samples = distribution.sample(num_samples=10000)
    assert np.isclose(np.linalg.norm(np.mean(samples, axis=0) - mu), 0, atol=0.1)
    assert np.isclose(np.std(samples[:, 0]), np.sqrt(sigma[0, 0]), atol=0.1)
    assert np.isclose(np.std(samples[:, 1]), np.sqrt(sigma[1, 1]), atol=0.1)


def test_gaussian2d_json():
    gauss_1 = murseco.utility.stats.Gaussian2D(np.array([4.1, -1.39]), np.eye(2) * 0.14)
    cache_path = murseco.utility.io.path_from_home_directory("test/cache/gaussian2d_test.json")
    gauss_1.to_json(cache_path)
    gauss_2 = murseco.utility.stats.Gaussian2D.from_json(cache_path)
    assert gauss_1.summary() == gauss_2.summary()
    assert gauss_1.pdf_at(0.0, 0.0) == gauss_2.pdf_at(0.0, 0.0)


@pytest.mark.parametrize(
    "mus, sigmas, weights, point, target",
    [
        (np.array([[1, 0], [0, 0]]), np.stack((np.eye(2), np.eye(2))), np.array([1, 0]), (0, 0), 0.096),
        (np.array([[0, 0], [0, 0]]), np.stack((np.eye(2) * 0.01, np.eye(2))), np.array([1, 0]), (10, 10), 0.0),
        (np.array([[0, 0], [0, 0]]), np.stack((np.diag([0.01, 10]), np.eye(2))), np.array([1, 0]), (0, 10), 0.0033),
    ],
)
def test_gmm2d_pdf_at(
    mus: np.ndarray, sigmas: np.ndarray, weights: np.ndarray, point: Tuple[float, float], target: float
):
    distribution = murseco.utility.stats.GMM2D(mus, sigmas, weights)
    assert np.isclose(distribution.pdf_at(point[0], point[1]), target, rtol=0.1)


@pytest.mark.parametrize(
    "mus, sigmas, weights",
    [(np.array([[4.1, -1.39], [2.1, -9.1]]), np.stack((np.eye(2) * 0.14, np.eye(2))), np.array([0.1, 1.0]))],
)
def test_gmm2d_json(mus: np.ndarray, sigmas: np.ndarray, weights: np.ndarray):
    gmm_1 = murseco.utility.stats.GMM2D(mus, sigmas, weights)
    cache_path = murseco.utility.io.path_from_home_directory("test/cache/gmm2d_test.json")
    gmm_1.to_json(cache_path)
    gmm_2 = murseco.utility.stats.GMM2D.from_json(cache_path)
    assert gmm_1.summary() == gmm_2.summary()
    assert gmm_1.pdf_at(0.0, 0.0) == gmm_2.pdf_at(0.0, 0.0)


@pytest.mark.parametrize(
    "mus, sigmas, weights",
    [
        (np.array([[4.1, -1.39], [2.1, -9.1]]), np.stack((np.eye(2) * 0.14, np.eye(2))), np.array([0.1, 1.0])),
        (np.array([[1.0, -9.3], [5, 5]]), np.stack((np.eye(2) * 0.1, np.eye(2) * 0.001)), np.array([0.1, 1.0])),
    ],
)
def test_gaussian2d_sample(mus: np.ndarray, sigmas: np.ndarray, weights: np.ndarray):
    # Idea: Multimodal Gaussian is sampled by choosing randomly with Gaussian to sample and then sample from this
    # Gaussian. Therefore a simple comparison between the samples mean and the mu vector will validate this approach.
    # Thus, the samples are clustered using k-means-algorithm, while the cluster centers should represent the
    # the mean of the distributions.
    np.random.seed(0)
    gmm = murseco.utility.stats.GMM2D(mus, sigmas, weights)
    samples = gmm.sample(2000)
    centers = scipy.cluster.vq.kmeans(samples, k_or_guess=gmm.num_modes)[0]
    assert np.isclose(np.linalg.norm(np.sort(centers, axis=0) - np.sort(mus, axis=0)), 0, atol=0.1)


def test_gmm2d_add():
    mus_1, sigmas_1, weights_1 = np.ones((2, 2)), np.stack((np.eye(2) * 0.14, np.eye(2))), np.ones(2)
    gmm_1 = murseco.utility.stats.GMM2D(mus_1, sigmas_1, weights_1)

    mus_2, sigmas_2, weights_2 = np.zeros((2, 2)), np.stack((np.eye(2) * 0.1, np.eye(2))), np.array([0.4, 0.2])
    gmm_2 = murseco.utility.stats.GMM2D(mus_2, sigmas_2, weights_2)

    gmm_3 = gmm_1 + gmm_2

    assert len(gmm_3.gaussians) == 4
    assert np.isclose(np.linalg.norm(gmm_3.weights - np.array([[0.25, 0.25, 0.33, 0.1667]])), 0, atol=0.1)
    assert np.array_equal(gmm_3.mus[0, :], np.array([1, 1]))
    assert np.array_equal(gmm_3.mus[1, :], np.array([1, 1]))
    assert np.array_equal(gmm_3.mus[2, :], np.array([0, 0]))
    assert np.array_equal(gmm_3.mus[3, :], np.array([0, 0]))
