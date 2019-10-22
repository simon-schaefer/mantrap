import numpy as np

from murseco.environment import Environment
from murseco.obstacle import AngularDTVObstacle


def vertical_fast(thorizon: int = 20, dt: float = 1.0) -> Environment:
    env_xaxis, env_yaxis = (-10, 10), (-10, 10)
    env = Environment(env_xaxis, env_yaxis, thorizon=thorizon, dt=dt)

    obs_position = np.array([2, -10])
    obs_mu = np.array([[0.0, 1.9]])
    obs_cov = np.array([np.diag([0.1, 0.1])])
    obs_w = np.ones(1)
    env.add_obstacle(AngularDTVObstacle, history=obs_position, mus=obs_mu, covariances=obs_cov, weights=obs_w)

    obs_position = np.array([4, -15])
    obs_mu = np.array([[0.0, 2.0], [0.02, 0.5]])
    obs_cov = np.array([np.diag([0.1, 0.1]), np.diag([0.2, 0.001])])
    obs_w = np.ones(2)
    env.add_obstacle(AngularDTVObstacle, history=obs_position, mus=obs_mu, covariances=obs_cov, weights=obs_w)

    return env
