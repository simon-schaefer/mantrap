import numpy as np

from murseco.obstacle import StaticDTVObstacle
from murseco.utility.array import rand_inv_pos_symmetric_matrix
from murseco.utility.io import path_from_home_directory


def test_ppdf():
    np.random.seed(0)
    mu, cov = np.random.rand(2), np.abs(np.diag(np.random.rand(2)))
    obstacle = StaticDTVObstacle(mu=mu, covariance=cov)
    tppdf = obstacle.tppdf(thorizon=20)

    for ppdf in tppdf:
        samples = ppdf.sample(1000)
        mean, cov_x, cov_y = np.mean(samples, axis=0), np.std(samples[:, 0], axis=0), np.std(samples[:, 1], axis=0)
        assert np.isclose(np.linalg.norm(mean - mu), 0, atol=0.1)
        assert np.isclose(np.linalg.norm(cov_x - np.sqrt(cov[0, 0])), 0, atol=0.1)
        assert np.isclose(np.linalg.norm(cov_y - np.sqrt(cov[1, 1])), 0, atol=0.1)


def test_samples():
    np.random.seed(0)
    mu, cov = np.random.rand(2), np.eye(2)
    obstacle = StaticDTVObstacle(mu=mu, covariance=cov)
    trajectory_samples = obstacle.trajectory_samples(thorizon=20, num_samples=200)

    mean_x = np.mean(trajectory_samples[:, :, 0])
    mean_y = np.mean(trajectory_samples[:, :, 1])
    assert np.isclose(np.linalg.norm(np.array([mean_x, mean_y]) - mu), 0, atol=0.1)


def test_json():
    obstacle_1 = StaticDTVObstacle(mu=np.zeros(2), covariance=rand_inv_pos_symmetric_matrix(2, 2))
    cache_path = path_from_home_directory("test/cache/staticobstacle_test.json")
    obstacle_1.to_json(cache_path)
    obstacle_2 = StaticDTVObstacle.from_json(cache_path)
    assert obstacle_1.summary() == obstacle_2.summary()
    assert obstacle_1.vpdf().pdf_at(0.2, 0) == obstacle_2.vpdf().pdf_at(0.2, 0)
