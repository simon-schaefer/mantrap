from typing import Tuple

import numpy as np
import pytest
import scipy.cluster.vq

from murseco.obstacle.cardinal import CardinalDiscreteTimeObstacle
from murseco.utility.arrayops import rand_invsymmpos
import murseco.utility.io
from murseco.utility.misc import cardinal_directions


def test_cardinalobstacle_initialization():
    pinit, pstep, tmax = np.array([1, 0]), 4, 10
    sigmas, weights = rand_invsymmpos(4, 2, 2), np.ones(4)
    obstacle = CardinalDiscreteTimeObstacle(pinit, pstep, tmax, sigmas, weights)
    assert obstacle is not None
    pstep = np.random.rand(4)
    obstacle = CardinalDiscreteTimeObstacle(pinit, pstep, tmax, sigmas, weights)
    assert obstacle is not None


@pytest.mark.parametrize(
    "pinit, point_init, target_init",
    [(np.array([1, 0]), (0, 0), 0.0), (np.array([1, 0]), (0.999, 0), 100), (np.array([1, 0]), (1.0, 0), 100)],
)  # np.inf = 100 (comp. definition of Point2D distribution)
def test_cardinalobstacle_pdf_init(pinit: np.ndarray, point_init: Tuple[float, float], target_init: float):
    pstep, tmax, sigmas, weights = 4, 10, rand_invsymmpos(4, 2, 2), np.ones(4)
    obstacle = CardinalDiscreteTimeObstacle(pinit, pstep, tmax, sigmas, weights)
    assert np.isclose(obstacle.pdf.pdf_at(point_init[0], point_init[1]), target_init, rtol=0.01)


def test_cardinalobstacle_tpdf():
    pinit, pstep, tmax, sigmas, weights = np.zeros(2), 2, 10, rand_invsymmpos(4, 2, 2) * 0.01, np.ones(4)
    obstacle = CardinalDiscreteTimeObstacle(pinit, pstep, tmax, sigmas, weights)
    samples = obstacle.tpdf[1].sample(2000)
    centers = scipy.cluster.vq.kmeans(samples, k_or_guess=4)[0]
    mus = pinit + pstep * cardinal_directions()
    assert np.isclose(np.linalg.norm(np.sort(centers, axis=0) - np.sort(mus, axis=0)), 0, atol=0.1)


def test_cardinalobstacle_json():
    pinit, pstep, tmax, sigmas, weights = np.zeros(2), 1, 10, rand_invsymmpos(4, 2, 2), np.ones(4)
    obstacle_1 = CardinalDiscreteTimeObstacle(pinit, pstep, tmax, sigmas, weights)
    cache_path = murseco.utility.io.path_from_home_directory("test/cache/cardinalobstacle_test.json")
    obstacle_1.to_json(cache_path)
    obstacle_2 = CardinalDiscreteTimeObstacle.from_json(cache_path)
    assert obstacle_1.summary() == obstacle_2.summary()
    assert obstacle_1.tpdf[1].pdf_at(0.0, 0.0) == obstacle_2.tpdf[1].pdf_at(0.0, 0.0)


