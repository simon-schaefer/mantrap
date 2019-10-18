import os
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

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


def plot_trajectory_samples(
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
    :argument fpath: path to store plot in.
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


def plot_tppdf(
    tppdf: List[np.ndarray],
    meshgrid: Tuple[np.ndarray, np.ndarray],
    dpath: str,
    rtrajectory: Union[np.ndarray, None] = None,
):
    """Plot pdf in position space in meshgrid and (optionally) the robot's trajectory over the full time horizon,
    plotting one plot per time-step.

    :argument tppdf: overall pdf in position space for each time-step.
    :argument meshgrid: (x, y) meshgrid which all the pdfs in tppdf are based on.
    :argument dpath: path to store directory in.
    :argument rtrajectory: robot trajectory in the given time-horizon.
    """
    if rtrajectory is not None:
        assert len(tppdf) == rtrajectory.shape[0] - 1, "length of tppdf (t >= 1) and robot's trajectory (t >= 0)"
    assert all([meshgrid[0].shape == ppdf.shape for ppdf in tppdf]), "x grid should have same shape as pdfs"
    assert all([meshgrid[1].shape == ppdf.shape for ppdf in tppdf]), "y grid should have same shape as pdfs"

    os.makedirs(dpath, exist_ok=True)

    for t, ppdf in enumerate(tppdf):
        fig, ax = plt.subplots()

        if rtrajectory is not None:
            ax.plot(rtrajectory[: t + 1, 0], rtrajectory[: t + 1, 1], "rx")

        color_mesh = ax.pcolormesh(meshgrid[0], meshgrid[1], ppdf, cmap="gist_earth")
        fig.colorbar(color_mesh, ax=ax)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.savefig(os.path.join(dpath, f"{t:04d}.png"))
        plt.close()


def plot_position_input_risk(positions: np.ndarray, inputs: np.ndarray, risk: np.ndarray, fpath: str):
    """Plot output of optimization, namely positions, input and risk values in separate plots over time-horizon.

    :argument positions: positions of agent (time-horizon, N).
    :argument inputs: control inputs (time-horizon, num_inputs).
    :argument risk: risk vector (time-horizon,).
    :argument fpath: path to store plot in.
    """

    thorizon = positions.shape[0]
    assert positions.shape[0] == risk.size, "time-horizon of positions and risk must be equal"
    assert positions.shape[0] == inputs.shape[0] + 1, "time-horizon of positions and inputs (+1) must be equal"

    fig, ax = plt.subplots(3, figsize=(10, 7))

    ax[0].plot(np.linspace(0, thorizon - 1, thorizon), positions[:, 0], label="x")
    ax[0].plot(np.linspace(0, thorizon - 1, thorizon), positions[:, 1], label="y")
    ax[0].set_xlabel("x_k")
    ax[0].set_ylabel("t_k")
    ax[0].legend()

    ax[1].plot(np.linspace(0, thorizon - 2, thorizon - 1), np.linalg.norm(inputs, axis=1), label="u")
    ax[1].set_xlabel("u_k")
    ax[1].set_ylabel("t_k")
    ax[1].legend()

    ax[2].plot(np.linspace(0, thorizon - 1, thorizon), risk, label="ux")
    ax[2].set_xlabel("r_k")
    ax[2].set_ylabel("t_k")
    ax[2].legend()

    plt.savefig(fpath)
    plt.close()
