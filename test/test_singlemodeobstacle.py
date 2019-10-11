import numpy as np

from murseco.obstacle.single_mode import SingleModeDiscreteTimeObstacle
from murseco.utility.arrayops import rand_invsymmpos
import murseco.utility.io


def test_singlemodeobstacle_iid():
    obstacle = SingleModeDiscreteTimeObstacle(covariance=rand_invsymmpos(2, 2))
    history_1, history_2 = np.random.rand(2, 2), np.random.rand(5, 2)
    history_1[-1, :] = history_2[-1, :]  # last position must be equivalent for mean
    pdf_1 = obstacle.pdf(history=history_1)
    pdf_2 = obstacle.pdf(history=history_2)
    assert pdf_1.summary() == pdf_2.summary()


def test_singlemodeobstacle_mean_progress():
    obstacle = SingleModeDiscreteTimeObstacle(covariance=rand_invsymmpos(2, 2))
    pdf_1 = obstacle.pdf(history=np.array([[0, 0], [1, 1]]))
    pdf_2 = obstacle.pdf(history=np.array([[0, 0], [-1, 5]]))
    assert np.array(pdf_1.mu != pdf_2.mu).all()


def test_singlemodeobstacle_json():
    obstacle_1 = SingleModeDiscreteTimeObstacle(history=np.zeros(2), covariance=rand_invsymmpos(2, 2))
    cache_path = murseco.utility.io.path_from_home_directory("test/cache/singlemodeobstacle_test.json")
    obstacle_1.to_json(cache_path)
    obstacle_2 = SingleModeDiscreteTimeObstacle.from_json(cache_path)
    assert obstacle_1.summary() == obstacle_2.summary()
    assert obstacle_1.pdf().pdf_at(0.2, 0) == obstacle_2.pdf().pdf_at(0.2, 0)
