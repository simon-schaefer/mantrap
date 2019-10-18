import numpy as np
import pytest
from scipy.cluster.vq import kmeans

from murseco.obstacle import AngularDTVObstacle
from murseco.utility.array import rand_invsymmpos
import murseco.utility.io


def test_angularobstacle_initialization():
    pinit = np.array([1, 0])
    obstacle = AngularDTVObstacle(pinit, mus=np.zeros((3, 2)), covariances=rand_invsymmpos(3, 2, 2), weights=np.ones(3))
    assert obstacle is not None
    obstacle = AngularDTVObstacle(
        pinit, mus=np.random.rand(4, 2), covariances=rand_invsymmpos(4, 2, 2), weights=np.ones(4)
    )
    assert obstacle is not None
    obstacle = AngularDTVObstacle()
    assert obstacle is not None


@pytest.mark.parametrize(
    "mus, covariances, weights",
    [(np.ones((1, 2)), np.eye(2) * 0.1, np.array([1.0])), (np.zeros(2), rand_invsymmpos(2, 2), np.ones(1))],
)
def test_angularobstacle_pdf(mus: np.ndarray, covariances: np.ndarray, weights: np.ndarray):
    np.random.seed(0)
    obstacle = AngularDTVObstacle(mus=mus, covariances=covariances, weights=weights)
    samples = obstacle.vpdf().sample(3000)
    center_expected = mus
    center = kmeans(samples, k_or_guess=weights.size)[0]
    assert np.isclose(np.linalg.norm(np.sort(center, axis=0) - np.sort(center_expected, axis=0)), 0, atol=0.1)


def test_angularobstacle_json():
    pinit, mus, covariances, weights = np.zeros(2), np.random.rand(3, 2), rand_invsymmpos(3, 2, 2), np.ones(3)
    obstacle_1 = AngularDTVObstacle(pinit, mus, covariances, weights)
    cache_path = murseco.utility.io.path_from_home_directory("test/cache/angularobstacle_test.json")
    obstacle_1.to_json(cache_path)
    obstacle_2 = AngularDTVObstacle.from_json(cache_path)
    assert obstacle_1.summary() == obstacle_2.summary()
    assert obstacle_1.vpdf().pdf_at(0, 0) == obstacle_2.vpdf().pdf_at(0, 0)
