import numpy as np
import pytest
import scipy.cluster.vq

from murseco.obstacle.tgmm import TGMMDiscreteTimeObstacle
from murseco.utility.arrayops import rand_invsymmpos
import murseco.utility.io


@pytest.mark.xfail(raises=AssertionError)
def test_tgmmobstacle_tmax():
    tmus = np.ones((1, 2, 2))
    tsigmas = rand_invsymmpos(1, 2, 2, 2)
    tweights = np.ones((1, 2))
    obstacle = TGMMDiscreteTimeObstacle(tmus, tsigmas, tweights)
    obstacle.forward()


def test_tgmmobstacle_one_mode():
    tmus = np.ones((1, 2, 2))
    tsigmas = rand_invsymmpos(1, 2, 2, 2)
    tweights = np.ones((1, 2))
    obstacle = TGMMDiscreteTimeObstacle(tmus, tsigmas, tweights)
    assert obstacle is not None


@pytest.mark.parametrize(
    "mus, sigmas, weights",
    [
        (np.array([[4.1, -1.38], [2.1, -9.1]]), np.stack((np.eye(2) * 0.14, np.eye(2))), np.array([0.1, 1.0])),
        (np.array([[1.0, -9.3], [5, 5]]), np.stack((np.eye(2) * 0.1, np.eye(2) * 0.001)), np.array([0.1, 1.0])),
    ],
)
def test_tgmmobstacle_pdf(mus, sigmas, weights):
    tmus = np.reshape(mus, (1, -1, 2))
    tsigmas = np.reshape(sigmas, (1, -1, 2, 2))
    tweights = np.reshape(weights, (1, -1))
    obstacle = TGMMDiscreteTimeObstacle(tmus, tsigmas, tweights)
    samples = obstacle.pdf.sample(2000)
    centers = scipy.cluster.vq.kmeans(samples, k_or_guess=obstacle.pdf.num_modes)[0]
    assert np.isclose(np.linalg.norm(np.sort(centers, axis=0) - np.sort(mus, axis=0)), 0, atol=0.1)


@pytest.mark.parametrize(
    "tmus, tsigmas, tweights", [(np.random.rand(2, 2, 2), rand_invsymmpos(2, 2, 2, 2), np.random.rand(2, 2))]
)
def test_tgmmobstacle_json(tmus: np.ndarray, tsigmas: np.ndarray, tweights: np.ndarray):
    gmmobstacle_1 = TGMMDiscreteTimeObstacle(tmus, tsigmas, tweights)
    cache_path = murseco.utility.io.path_from_home_directory("test/cache/gmmobstacle_test.json")
    gmmobstacle_1.to_json(cache_path)
    gmmobstacle_2 = TGMMDiscreteTimeObstacle.from_json(cache_path)
    assert gmmobstacle_1.summary() == gmmobstacle_2.summary()
    assert gmmobstacle_1.pdf.pdf_at(0.0, 0.0) == gmmobstacle_2.pdf.pdf_at(0.0, 0.0)
