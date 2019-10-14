import logging

import numpy as np

from murseco.environment import Environment
from murseco.obstacle import SingleModeDTVObstacle
from murseco.robot import IntegratorDTRobot
from murseco.utility.io import path_from_home_directory
from murseco.utility.misc import random_string
from murseco.utility.visualization import plot_env_samples


def main():
    scenario_label = random_string(5)
    logging.info(f"Creating scenario with label = {scenario_label}")
    # Build general environment.
    env_xaxis, env_yaxis = (-10, 10), (-10, 10)
    env = Environment(env_xaxis, env_yaxis)

    # Add dynamic obstacles to environment.
    sm_obs_1_pinit, sm_obs_1_vmu, sm_obs_1_vcov = np.array([5, -5]), np.array([0, 1]), np.diag([0.2, 1e-4])
    env.add_obstacle(SingleModeDTVObstacle(history=sm_obs_1_pinit, mu=sm_obs_1_vmu, covariance=sm_obs_1_vcov))

    # Add robot to environment.
    robot_pinit, robot_thorizon = np.array([-5, 0]), 10
    env.add_robot(IntegratorDTRobot(position=robot_pinit, thorizon=robot_thorizon))

    # Store and visualize environment.
    env.to_json(path_from_home_directory(f"config/{scenario_label}.json"))
    plot_env_samples(env, path_from_home_directory(f"config/{scenario_label}.png"))
    logging.info(f"Saved scenario json and initial scene at config directory")


if __name__ == '__main__':
    main()
