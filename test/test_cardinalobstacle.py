import numpy as np

from murseco.obstacle.cardinal import CardinalDiscreteTimeObstacle
from murseco.utility.arrayops import rand_invsymmpos
import murseco.utility.io


def test_cardinalobstacle_initialization():
    pinit, sigmas, weights = np.array([1, 0]), rand_invsymmpos(4, 2, 2), np.ones(4)
    obstacle = CardinalDiscreteTimeObstacle(pinit, 4, sigmas, weights)
    assert obstacle is not None
    obstacle = CardinalDiscreteTimeObstacle(pinit, np.random.rand(4), sigmas, weights)
    assert obstacle is not None


def test_cardinalobstacle_json():
    pinit, velocity, sigmas, weights = np.zeros(2), 1.2, rand_invsymmpos(4, 2, 2), np.ones(4)
    obstacle_1 = CardinalDiscreteTimeObstacle(pinit, velocity, sigmas, weights)
    cache_path = murseco.utility.io.path_from_home_directory("test/cache/cardinalobstacle_test.json")
    obstacle_1.to_json(cache_path)
    obstacle_2 = CardinalDiscreteTimeObstacle.from_json(cache_path)
    assert obstacle_1.summary() == obstacle_2.summary()
    assert obstacle_1.pdf().pdf_at(0, 0) == obstacle_2.pdf().pdf_at(0, 0)
