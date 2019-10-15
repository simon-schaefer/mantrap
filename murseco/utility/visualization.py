import os
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from murseco.environment.environment import Environment
from murseco.utility.misc import MATPLOTLIB_MARKERS
from murseco.utility.stats import Distribution2D


def plot_pdf2d(
    distribution: Distribution2D,
    xaxis: Tuple[float, float],
    fpath: str,
    yaxis: Union[Tuple[float, float], None] = None,
    num_points: int = 500,
):
    """Plot 2D PDF (probability density function) in mesh grid with the given axes.
    In order to plot the PDF the axes are split up in a constant number of points, i.e. the resolution varies with
    the expansion of the axes. However the default number of points is large enough to draw a fairly accurate
    representation of the distribution while being efficient. If just one axis is given, the other axis is assumed
    to be the same as the given axis.

    :argument distribution: 2D distribution as subclass of Distribution2D class in utility.stats.
    :argument xaxis: range of x axis as (lower bound, upper bound).
    :argument fpath: path to store directory in.
    :argument yaxis: range of y axis as (lower bound, upper bound), assumed to be equal to xaxis if not stated.
    :argument num_points: number of resolution points for axis sampling.
    """
    fig, ax = plt.subplots()

    yaxis = xaxis if yaxis is None else yaxis
    num_points = int(num_points)
    x, y = np.meshgrid(np.linspace(xaxis[0], xaxis[1], num_points), np.linspace(yaxis[0], yaxis[1], num_points))
    pdf = distribution.pdf_at(x, y)
    color_mesh = ax.pcolormesh(x, y, pdf, cmap="gist_earth")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(color_mesh)

    plt.savefig(fpath)
    plt.close()


def plot_env_samples(
    otrajectory_samples: List[np.ndarray],
    ohistories: List[np.ndarray],
    ocolors: List[str],
    xaxis: Tuple[float, float],
    fpath: str,
    rtrajectory: Union[np.ndarray, None] = None,
    yaxis: Union[Tuple[float, float], None] = None,
):
    """Plot N = num_samples possible trajectories for all the dynamic obstacles in the scene as well as the trajectory
    of the robot, both the obstacle's trajectory samples as well as the robot's planned trajectory for k = planning_
    horizon of the robot time-steps.

    :argument otrajectory_samples: sampled trajectories for each obstacle (obstacle_i, num_modes, N, time-horizon, 2).
    :argument ohistories: history path of each obstacle (obstacle_i, num_history_points, 2).
    :argument ocolors: color identifier of each obstacle
    :argument xaxis: range of x axis as (lower bound, upper bound).
    :argument fpath: path to store directory in.
    :argument yaxis: range of y axis as (lower bound, upper bound), assumed to be equal to xaxis if not stated.
    :argument rtrajectory: robot trajectory in the given time-horizon.
    """
    assert len(otrajectory_samples) == len(ohistories), "unequal number of obstacles in histories and samples"
    assert len(otrajectory_samples) == len(ocolors), "unequal number of obstacles in colors and samples"

    fig, ax = plt.subplots()
    yaxis = xaxis if yaxis is None else yaxis

    if rtrajectory is not None:
        ax.plot(rtrajectory[:, 0], rtrajectory[:, 1], "rx")

    for trajectory, history, color in zip(otrajectory_samples, ohistories, ocolors):
        for m in range(trajectory.shape[0]):
            mode_marker = np.random.choice(MATPLOTLIB_MARKERS)
            for i in range(trajectory.shape[1]):
                ax.plot(trajectory[m, i, :, 0], trajectory[m, i, :, 1], color + mode_marker)
            ax.plot(history[:, 0], history[:, 1], color + "o")

    ax.set_xlim(xaxis)
    ax.set_ylim(yaxis)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.savefig(fpath)
    plt.close()


def plot_env_tppdf(
    tppdf: List[np.ndarray],
    meshgrid: Tuple[np.ndarray, np.ndarray],
    dir_path: str,
    rtrajectory: Union[np.ndarray, None] = None,
):
    """Plot pdf in position space in meshgrid and (optionally) the robot's trajectory over the full time horizon,
    plotting one plot per time-step.

    :argument tppdf: overall pdf in position space for each time-step.
    :argument meshgrid: (x, y) meshgrid which all the pdfs in tppdf are based on.
    :argument dir_path: path to store directory in.
    :argument rtrajectory: robot trajectory in the given time-horizon.
    """
    if rtrajectory is not None:
        assert len(tppdf) == rtrajectory.shape[0], "length of tppdf and robot's trajectory should be equal"
    assert all([meshgrid[0].shape == ppdf.shape for ppdf in tppdf]), "x grid should have same shape as pdfs"
    assert all([meshgrid[1].shape == ppdf.shape for ppdf in tppdf]), "y grid should have same shape as pdfs"

    os.makedirs(dir_path, exist_ok=False)

    for t, ppdf in enumerate(tppdf):
        fig, ax = plt.subplots()

        if rtrajectory is not None:
            ax.plot(rtrajectory[t, 0], rtrajectory[t, 1], "rx")

        color_mesh = ax.pcolormesh(meshgrid[0], meshgrid[1], ppdf, cmap="gist_earth")
        fig.colorbar(color_mesh, ax=ax)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.savefig(os.path.join(dir_path, f"{t:04d}.png"))
        plt.close()
