from datetime import datetime
import os
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from murseco.environment.environment import Environment
from murseco.utility.stats import Distribution2D


def plot_pdf2d(
    fig: plt.Figure,
    ax: plt.Axes,
    distribution: Distribution2D,
    xaxis: Tuple[float, float],
    yaxis: Union[Tuple[float, float], None] = None,
    num_points: int = 500,
):
    """Plot 2D PDF (probability density function) in mesh grid with the given axes.
    In order to plot the PDF the axes are split up in a constant number of points, i.e. the resolution varies with
    the expansion of the axes. However the default number of points is large enough to draw a fairly accurate
    representation of the distribution while being efficient. If just one axis is given, the other axis is assumed
    to be the same as the given axis.

    :argument fig: matplotlib figure to draw in.
    :argument ax: matplotlib axis to draw in.
    :argument distribution: 2D distribution as subclass of Distribution2D class in utility.stats.
    :argument xaxis: range of x axis as (lower bound, upper bound).
    :argument yaxis: range of y axis as (lower bound, upper bound), assumed to be equal to xaxis if not stated.
    :argument num_points: number of resolution points for axis sampling.
    """
    yaxis = xaxis if yaxis is None else yaxis
    num_points = int(num_points)
    x, y = np.meshgrid(np.linspace(xaxis[0], xaxis[1], num_points), np.linspace(yaxis[0], yaxis[1], num_points))
    pdf = distribution.pdf_at(x, y)
    color_mesh = ax.pcolormesh(x, y, pdf, cmap="gist_earth")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(color_mesh)


def plot_env_samples(env: Environment, fpath: str, num_samples: int = 10, num_points: int = 500):
    """Plot 2D environment including the attached actors like obstacles or robots for given time-step.

    :argument env: environment object to plot.
    :argument fpath: path to store directory in.
    :argument num_samples: number of trajectories to sample.
    :argument num_points: number of resolution points for axis sampling.
    """
    fig, ax = plt.subplots()
    # num_points = int(num_points)
    # x_min, x_max, y_min, y_max = env.xaxis[0], env.xaxis[1], env.yaxis[0], env.yaxis[1]
    # x, y = np.meshgrid(np.linspace(x_min, x_max, num_points), np.linspace(y_min, y_max, num_points))
    time_horizon = 10

    robot = env.robot
    if robot is not None:
        trajectory = robot.trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], "r-")
        time_horizon = robot.planning_horizon

    for obstacle in env.obstacles:
        trajectory_samples = obstacle.trajectory_samples(time_horizon, num_samples=num_samples)
        for i in range(trajectory_samples.shape[0]):
            ax.plot(trajectory_samples[i, :, 0], trajectory_samples[i, :, 1], obstacle.color + "--")
        history = obstacle.history
        ax.plot(history[:, 0], history[:, 1], "x")

    # pdf = sum([o.pdf().pdf_at(x, y) for o in env.obstacles])
    # color_mesh = ax.pcolormesh(x, y, pdf, cmap="gist_earth")
    # fig.colorbar(color_mesh, ax=ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.savefig(fpath)
    plt.close()
