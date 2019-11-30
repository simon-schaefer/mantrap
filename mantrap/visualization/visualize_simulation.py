import os

import matplotlib.pyplot as plt
import numpy as np

import mantrap.constants
from mantrap.utility.io import datetime_name, path_from_home_directory
from mantrap.simulation.abstract import Simulation


def plot_scene(
    sim: Simulation,
    ego_trajectory: np.ndarray = None,
    t_horizon: int = mantrap.constants.t_horizon_default,
    output_dir: str = path_from_home_directory(f"outs/{datetime_name()}"),
    image_tag: str = "0",
):
    """Visualize simulation scene using matplotlib library.
    Thereby the ados as well as the ego in at the current time are plotted while their future trajectories
    and their state histories are indicated. Their orientation is shown using an arrow pointing in their direction
    of orientation.
    :param sim: simulation object to plot (abstract class contains all required methods, so sim has to inherit from it).
    :param ego_trajectory: planned ego trajectory (t_horizon, 5).
    :param t_horizon: prediction horizon (number of time-steps of length dt).
    :param output_dir: output directory file path.
    :param image_tag: image name (tag for current time-step).
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # Predict ado trajectories. In case the prediction is deterministic (single trajectory for each ado), reshape
    # the resulting array, for a unified format.
    trajectories = sim.predict(t_horizon, ego_trajectory=ego_trajectory)
    if len(trajectories.shape) == 2:
        trajectories = np.expand_dims(trajectories, axis=0)

    # Plot ados.
    for ado in sim.ados:
        ado_arrow_length = ado.speed / mantrap.constants.sim_speed_max * 0.5
        ax = _add_agent_representation(ado.pose, color=ado.color, ax=ax, arrow_length=ado_arrow_length)
        ax = _add_history(ado.history, color=ado.color, ax=ax)
        for i in range(trajectories.shape[0]):
            ax = _add_trajectory(trajectories[i, :, :], color=ado.color, ax=ax)

    # Plot ego.
    if sim.ego is not None:
        ego_color = np.array([0, 0, 1.0])
        ax = _add_agent_representation(sim.ego.pose, color=ego_color, ax=ax)
        ax = _add_history(sim.ego.history, color=ego_color, ax=ax)
        if ego_trajectory is not None:
            ax = _add_trajectory(ego_trajectory, color=ego_color, ax=ax)

    # Plot labels, limits and grid.
    x_axis, y_axis = sim.axes
    plt.xlabel("x [m]")
    plt.xlim(x_axis)
    plt.ylabel("y [m]")
    plt.ylim(y_axis)
    plt.minorticks_on()
    plt.grid(which="minor", alpha=0.2)
    plt.grid(which="major", alpha=0.5)

    # Save and close plot.
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{image_tag}.png"))
    plt.close()


def _add_agent_representation(pose: np.ndarray, color: np.ndarray, ax: plt.Axes, arrow_length: float = 0.5):
    assert pose.size == 3, "pose must be 3D (x, y, theta)"

    ado_circle = plt.Circle((pose[0], pose[1]), 0.1, color=color, clip_on=True)
    ax.add_artist(ado_circle)

    rot = np.array([[np.cos(pose[2]), -np.sin(pose[2])], [np.sin(pose[2]), np.cos(pose[2])]])
    darrow = rot.dot(np.array([1, 0])) * arrow_length
    head_width = max(0.02, arrow_length / 10)
    plt.arrow(pose[0], pose[1], darrow[0], darrow[1], head_width=head_width, head_length=0.1, fc="k", ec="k")
    return ax


def _add_trajectory(trajectory: np.ndarray, color: np.ndarray, ax: plt.Axes):
    assert len(trajectory.shape) == 2, "trajectory must have shape (N, state_length)"
    ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linestyle="-", linewidth=0.6, alpha=1.0)
    return ax


def _add_history(history: np.ndarray, color: np.ndarray, ax: plt.Axes):
    assert len(history.shape) == 2, "history must have shape (M, state_length)"
    ax.plot(history[:, 0], history[:, 1], color=color, linestyle="-.", linewidth=0.6, alpha=0.6)
    return ax
