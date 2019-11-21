from datetime import datetime
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_environment(
    ados,  # List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ego,  # Tuple[np.ndarray, np.ndarray, np.ndarray],
    xaxis,  # Tuple[float, float],
    yaxis,  # Tuple[float, float],
    output_dir,  # str
):
    """Visualize environment using matplotlib.pyplot library. The environment is two-dimensional, has bounds
    `xaxis` and `yaxis` (min, max). It contains agents, which can be seperated in the ados ("obstacles") and
    the ego. Both have some current 2D pose (x, y, theta), history (previous poses) and future trajectories
    (next poses), both history and future stamped with timestamps as a 4D vector (x, y, theta, t). The future
    trajectories of the ados thereby is uncertain, so there could be multiple trajectories given.

    @param ados: list of ados, each having a current pose, history and (multiple) future trajectories.
    @param ego: ego having current pose, history and (one) future trajectory.
    @param xaxis: environment expansion in x direction (min, max).
    @param yaxis: environment expansion in y direction (min, max).
    @param output_dir: output directory for save files.
    """
    for ado in ados:
        assert ado[0].size == 3, "ado pose must be 3D (x, y, theta)"
        assert len(ado[1].shape) == 2 and ado[1].size % 4 == 0, "ado history must be of shape (N, 4)"
        assert len(ado[2].shape) == 3 and ado[2].shape[2] == 4, "ado trajectories must be of shape (M, thorizon, 4)"
    assert ego[0].size == 3, "ego pose must be 3D (x, y, theta)"
    if ego[1].size > 0:
        assert len(ego[1].shape) == 2 and ego[1].size % 4 == 0, "ego history must be of shape (N, 4)"
    if ego[2].size > 0:
        assert len(ego[2].shape) == 2 and ego[1].size % 4 == 0, "ego trajectory must be of shape (thorizon, 4)"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot ados.
    for ado in ados:
        ax = _add_agent_representation(ado[0], color='r', ax=ax)
        if ado[1].size > 0:
            ax = _add_history(ado[1], color='r', ax=ax)
        if ado[2].size > 0:
            for k in range(ado[2].shape[0]):
                ax = _add_trajectory(ado[2][k, :, :], color='r', ax=ax)

    # Plot ego.
    ax = _add_agent_representation(ego[0], color='b', ax=ax)
    if ego[1].size > 0:
        ax = _add_history(ego[1], color='b', ax=ax)
    if ego[2].size > 0:
        ax = _add_trajectory(ego[2], color='b', ax=ax)

    # Plot labels, limits and grid.
    plt.xlabel("x [m]")
    plt.xlim(xaxis)
    plt.ylabel("y [m]")
    plt.ylim(yaxis)
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)

    # Save plot.
    plt.savefig(os.path.join(output_dir, "0.png"))
    plt.close()


def _add_agent_representation(pose,  # np.ndarray,
                              color,  # str
                              ax,  # plt.Axes
                              arrow_length=0.5,  # float
):
    assert pose.size == 3, "pose must be 3D (x, y, theta)"

    ado_circle = plt.Circle((pose[0], pose[1]), 0.1, color=color, clip_on=True)
    ax.add_artist(ado_circle)

    rot = np.array([[np.cos(pose[2]), np.sin(pose[2])], [-np.sin(pose[2]), np.cos(pose[2])]])
    darrow = rot.dot(np.array([1, 0])) * arrow_length
    plt.arrow(pose[0], pose[1], darrow[0], darrow[1], head_width=0.05, head_length=0.1, fc='k', ec='k')
    return ax


def _add_trajectory(trajectory,  # np.ndarray
                    color,  # str
                    ax,  # plt.Axes
):
    ax.plot(trajectory[:, 0], trajectory[:, 1], color + "-", linewidth=0.2, alpha=0.6)
    return ax


def _add_history(history,  # np.ndarray
                 color,  # str
                 ax,  # plt.Axes
):
    ax.plot(history[:, 0], history[:, 1], color + "--", linewidth=0.2, alpha=0.6)
    return ax
