import logging
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

import mantrap.constants
from mantrap.utility.shaping import check_ego_trajectory, check_ado_trajectories, extract_ado_trajectories


def plot_scene(
    ax: plt.Axes,
    t: int,
    ado_trajectories: np.ndarray,
    ado_colors: List[np.ndarray],
    ado_ids: List[str],
    ego_trajectory: np.ndarray = None,
    ado_trajectories_wo: np.ndarray = None,
    preview_horizon: int = mantrap.constants.visualization_preview_horizon,
    axes: Tuple[Tuple[float, float], Tuple[float, float]] = (
        mantrap.constants.sim_x_axis_default,
        mantrap.constants.sim_y_axis_default,
    ),
):
    """Visualize simulation scene using matplotlib library.
    Thereby the ados as well as the ego in at the current time are plotted while their future trajectories
    and their state histories are indicated. Their orientation is shown using an arrow pointing in their direction
    of orientation.
    :param t: time-step to plot.
    :param ax: matplotlib axis to draw in.
    :param ego_trajectory: planned ego trajectory (t_horizon, 6).
    :param ado_trajectories: ado trajectories (num_ados, num_samples, t_horizon, 6).
    :param ado_colors: color identifier for each ado (num_ados).
    :param ado_ids: string identifier for each ado (num_ados).
    :param ado_trajectories_wo: ado trajectories without ego interaction (num_ados, num_samples=1, t_horizon, 6).
    :param preview_horizon: trajectory preview time horizon (maximal value).
    :param axes: position space dimensions [m].
    """
    assert check_ado_trajectories(ado_trajectories=ado_trajectories)
    num_ados, num_modes, t_horizon = extract_ado_trajectories(ado_trajectories)
    assert len(ado_colors) == num_ados, "ado colors must be consistent with trajectories"
    ado_ids = [None] * num_ados if ado_ids is None else ado_ids
    assert len(ado_ids) == num_ados, "ado ids must be consistent with trajectories"
    if ego_trajectory is not None:
        assert check_ego_trajectory(ego_trajectory=ego_trajectory, t_horizon=t_horizon)
    if ado_trajectories_wo is not None:
        assert check_ado_trajectories(ado_trajectories=ado_trajectories_wo, num_modes=1, num_ados=num_ados)
    # logging.debug(f"Plotting scene with {num_ados} ados having {num_modes} modes for T = {t_horizon}")

    # Plot ados.
    ado_preview = min(preview_horizon, t_horizon - t)
    for ado_i in range(num_ados):
        ado_pose = ado_trajectories[ado_i, 0, t, 0:3]
        ado_velocity = ado_trajectories[ado_i, 0, t, 4:6]
        ado_arrow_length = np.linalg.norm(ado_velocity) / mantrap.constants.sim_speed_max * 0.5
        ado_color = ado_colors[ado_i]
        ado_id = ado_ids[ado_i]
        ado_history = ado_trajectories[ado_i, 0, :t, 0:2]

        ax = _add_agent_representation(ado_pose, color=ado_color, name=ado_id, ax=ax, arrow_length=ado_arrow_length)
        ax = _add_history(ado_history, color=ado_color, ax=ax)
        for mode_i in range(num_modes):
            ax = _add_trajectory(ado_trajectories[ado_i, mode_i, t : t + ado_preview, 0:2], color=ado_color, ax=ax)
        if ado_trajectories_wo is not None:
            ax = _add_wo_trajectory(ado_trajectories_wo[ado_i, 0, :t, 0:2], color=ado_color, ax=ax)

    # Plot ego.
    if ego_trajectory is not None:
        ego_pose = ego_trajectory[t, 0:3]
        ego_color = np.array([0, 0, 1.0])
        ego_history = ego_trajectory[:t, 0:2]
        ego_preview = min(preview_horizon, ego_trajectory.shape[0] - t)
        ego_planned = ego_trajectory[t : t + ego_preview, 0:2]

        ax = _add_agent_representation(ego_pose, color=ego_color, name="ego", ax=ax)
        ax = _add_history(ego_history, color=ego_color, ax=ax)
        ax = _add_trajectory(ego_planned, color=ego_color, ax=ax)

    # Plot labels, limits and grid.
    x_axis, y_axis = axes
    ax.set_xlim(xmin=x_axis[0], xmax=x_axis[1])
    ax.set_ylim(ymin=y_axis[0], ymax=y_axis[1])
    ax.tick_params(axis="both", which="major", labelsize=5)
    ax.grid(which="minor", alpha=0.1)
    ax.grid(which="major", alpha=0.3)
    return ax


def _add_agent_representation(
    pose: np.ndarray, color: np.ndarray, name: Union[str, None], ax: plt.Axes, arrow_length: float = 0.5
):
    assert pose.size == 3, "pose must be 3D (x, y, theta)"

    ado_circle = plt.Circle((pose[0], pose[1]), mantrap.constants.visualization_agent_radius, color=color, clip_on=True)
    ax.add_artist(ado_circle)

    # Add agent id description.
    if id is not None:
        ax.text(pose[0], pose[1], name, fontsize=8)

    # Add arrow for orientation and speed.
    rot = np.array([[np.cos(pose[2]), -np.sin(pose[2])], [np.sin(pose[2]), np.cos(pose[2])]])
    darrow = rot.dot(np.array([1, 0])) * arrow_length
    head_width = max(0.02, arrow_length / 10)
    plt.arrow(pose[0], pose[1], darrow[0], darrow[1], head_width=head_width, head_length=0.1, fc="k", ec="k")
    return ax


def _add_trajectory(trajectory: np.ndarray, color: np.ndarray, ax: plt.Axes):
    assert len(trajectory.shape) == 2, "trajectory must have shape (N, state_length)"
    ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linestyle="-", linewidth=0.6, alpha=1.0)
    return ax


def _add_wo_trajectory(trajectory: np.ndarray, color: np.ndarray, ax: plt.Axes):
    assert len(trajectory.shape) == 2, "trajectory must have shape (N, state_length)"
    ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linestyle=":", linewidth=0.6, alpha=1.0)
    return ax


def _add_history(history: np.ndarray, color: np.ndarray, ax: plt.Axes):
    assert len(history.shape) == 2, "history must have shape (M, state_length)"
    ax.plot(history[:, 0], history[:, 1], color=color, linestyle="-.", linewidth=0.6, alpha=0.8)
    return ax
