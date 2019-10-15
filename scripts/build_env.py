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
    env = Environment(env_xaxis, env_yaxis, thorizon=10, dt=1.0)

    # Add dynamic obstacles to environment.
    obs_1_pinit = np.array([5, -5])
    obs_1_vmu = np.array([0, 1])
    obs_1_vcov = np.diag([0.2, 1e-4])
    env.add_obstacle(SingleModeDTVObstacle, history=obs_1_pinit, mu=obs_1_vmu, covariance=obs_1_vcov)

    obs_2_p = np.array([-4, -3])
    obs_2_mu = np.array([[0.75, 0], [0.5, 0.02]])
    obs_2_cov = np.array([np.diag([0.001, 0.001]), np.diag([0.001, 0.2])])
    obs2_w = np.ones(2)
    env.add_obstacle(AngularDTVObstacle, history=obs_2_p, mus=obs_2_mu, covariances=obs_2_cov, weights=obs2_w)

    # Add robot to environment.
    robot_pinit = np.array([-5, 0])
    env.add_robot(IntegratorDTRobot, position=robot_pinit)
    logging.info(f"Built environment with {len(env.obstacles)} dynamic obstacles")

    # Store and visualize environment (trajectory samples, tppdf).
    env.to_json(path_from_home_directory(f"config/{scenario_label}.json"))

    otrajectory_samples = env.generate_trajectory_samples()
    ohistories = [o.history for o in env.obstacles]
    ocolors = [o.color for o in env.obstacles]
    tppdf, meshgrid = env.tppdf()
    rtrajectory = env.robot.trajectory()

    plot_env_samples(
        otrajectory_samples,
        ohistories,
        ocolors,
        xaxis=env.xaxis,
        fpath=path_from_home_directory(f"config/{scenario_label}.png"),
        rtrajectory=rtrajectory,
    )
    plot_env_tppdf(
        tppdf, meshgrid, dir_path=path_from_home_directory(f"config/{scenario_label}"), rtrajectory=rtrajectory
    )
    logging.info(f"Saved scenario json and initial scene at config directory")


if __name__ == "__main__":
    main()
