import logging
import sys

import numpy as np

from murseco.environment import Environment
from murseco.obstacle import SingleModeDTVObstacle, AngularDTVObstacle
from murseco.robot import IntegratorDTRobot
from murseco.utility.io import path_from_home_directory
from murseco.utility.visualization import plot_env_samples, plot_env_tppdf


def main():
    assert len(sys.argv) > 1, "invalid script call, use `python3 build_env.py 'scenario_label'`"
    scenario_label = sys.argv[1]
    logging.info(f"Creating scenario with label = {scenario_label}")
    # Build general environment.
    env_xaxis, env_yaxis = (-10, 10), (-10, 10)
    env = Environment(env_xaxis, env_yaxis)

    # Add dynamic obstacles to environment.
    obs_1_pinit = np.array([5, -5])
    obs_1_vmu = np.array([0, 1])
    obs_1_vcov = np.diag([0.2, 1e-4])
    env.add_obstacle(SingleModeDTVObstacle(history=obs_1_pinit, mu=obs_1_vmu, covariance=obs_1_vcov))

    obs_2_p = np.array([-4, -3])
    obs_2_mu = np.array([[0.75, 0], [0.5, 0.02]])
    obs_2_cov = np.array([np.diag([0.001, 0.001]), np.diag([0.001, 0.2])])
    obs2_w = np.ones(2)
    env.add_obstacle(AngularDTVObstacle(history=obs_2_p, mus=obs_2_mu, covariances=obs_2_cov, weights=obs2_w))

    # Add robot to environment.
    robot_pinit, robot_thorizon = np.array([-5, 0]), 10
    env.add_robot(IntegratorDTRobot(position=robot_pinit, thorizon=robot_thorizon))

    # Store and visualize environment (trajectory samples, tppdf).
    env.to_json(path_from_home_directory(f"config/{scenario_label}.json"))
    plot_env_samples(env, path_from_home_directory(f"config/{scenario_label}.png"))
    plot_env_tppdf(env, path_from_home_directory(f"config/{scenario_label}"))
    logging.info(f"Saved scenario json and initial scene at config directory")


if __name__ == '__main__':
    main()
