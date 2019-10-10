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


def plot_env_at_time(fig: plt.Figure, ax: plt.Axes, env: Environment, time_step: int = 0, num_points: int = 500):
    """Plot 2D environment including the attached actors like obstacles or robots for given time-step.

    :argument fig: matplotlib figure to draw in.
    :argument ax: matplotlib axis to draw in.
    :argument env: environment object to plot.
    :argument time_step: time-step to plot (0....tmax, default = 0).
    :argument num_points: number of resolution points for axis sampling.
    """
    num_points = int(num_points)
    x, y = np.meshgrid(
        np.linspace(env.xaxis[0], env.xaxis[1], num_points), np.linspace(env.yaxis[0], env.yaxis[1], num_points)
    )

    pdf = np.zeros_like(x)
    for obstacle in env.obstacles:
        pdf += obstacle.tpdf[time_step].pdf_at(x, y)

    robot = env.robot
    if robot is not None:
        trajectory = robot.trajectory
        ax.plot(trajectory[:time_step + 1, 0], trajectory[:time_step + 1, 1], "-")

    color_mesh = ax.pcolormesh(x, y, pdf, cmap="gist_earth")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(color_mesh, ax=ax)


def plot_env_all_times(env: Environment, fpath: str, num_points: int = 500):
    """Plot 2D environment including the attached actors like obstacles or robots over all times in horizon (0...tmax).
    Therefore create one plot for each time-step, saving all in a created directory.

    :argument env: environment object to plot.
    :argument fpath: path to store directory in.
    :argument num_points: number of resolution points for axis sampling.
    """
    dirname = os.path.join(fpath, datetime.now().strftime("%Y_%m_%d/%H_%M_%S"))
    os.makedirs(dirname, exist_ok=True)

    for tn in range(env.tmax):
        fig, ax = plt.subplots()
        plot_env_at_time(fig, ax, env, time_step=tn, num_points=num_points)
        plt.savefig(os.path.join(dirname, f"{tn}.png"))
        plt.close()
