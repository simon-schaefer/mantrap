import numpy as np
import pytest
from scipy.cluster.vq import kmeans

from murseco.obstacle.cardinal import CardinalDiscreteTimeObstacle
from murseco.utility.arrayops import rand_invsymmpos
import murseco.utility.io
from murseco.utility.misc import cardinal_directions


def test_cardinalobstacle_initialization():
    pinit, sigmas, weights = np.array([1, 0]), rand_invsymmpos(4, 2, 2), np.ones(4)
    obstacle = CardinalDiscreteTimeObstacle(pinit, 4, sigmas, weights)
    assert obstacle is not None
    obstacle = CardinalDiscreteTimeObstacle(pinit, np.random.rand(4), sigmas, weights)
    assert obstacle is not None


@pytest.mark.parametrize(
    "velocity, sigmas, weights",
    [(4, np.array([np.eye(2)] * 4), np.array([0.8, 0.9, 0.9, 0.85])),
     (10, rand_invsymmpos(4, 2, 2), np.ones(4))]
)
def test_cardinalobstacle_pdf(velocity: float, sigmas: np.ndarray, weights: np.ndarray):
    np.random.seed(0)
    obstacle = CardinalDiscreteTimeObstacle(velocity=velocity, sigmas=sigmas, weights=weights)
    samples = obstacle.pdf().sample(3000)
    center_expected = np.array([obstacle.position + velocity * cardinal_directions()[i, :] for i in range(4)])
    center = kmeans(samples, k_or_guess=4)[0]
    assert np.isclose(np.linalg.norm(np.sort(center, axis=0) - np.sort(center_expected, axis=0)), 0, atol=0.1)


def test_cardinalobstacle_json():
    pinit, velocity, sigmas, weights = np.zeros(2), 1.2, rand_invsymmpos(4, 2, 2), np.ones(4)
    obstacle_1 = CardinalDiscreteTimeObstacle(pinit, velocity, sigmas, weights)
    cache_path = murseco.utility.io.path_from_home_directory("test/cache/cardinalobstacle_test.json")
    obstacle_1.to_json(cache_path)
    obstacle_2 = CardinalDiscreteTimeObstacle.from_json(cache_path)
    assert obstacle_1.summary() == obstacle_2.summary()
    assert obstacle_1.pdf().pdf_at(0, 0) == obstacle_2.pdf().pdf_at(0, 0)
