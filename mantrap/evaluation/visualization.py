import os
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import torch

import mantrap.constants
from mantrap.utility.shaping import check_ego_trajectory, check_trajectories, extract_ado_trajectories, check_state


def picture_opus(
    file_path: str,
    ado_trajectories: torch.Tensor,
    ado_colors: List[List[float]],
    ado_ids: List[str],
    ego_trajectory: torch.Tensor = None,
    ado_trajectories_wo: torch.Tensor = None,
    t_preview: int = mantrap.constants.visualization_preview_horizon,
    axes: Tuple[Tuple[float, float], Tuple[float, float]] = (
        mantrap.constants.sim_x_axis_default,
        mantrap.constants.sim_y_axis_default,
    ),
):
    _, _, t_horizon = extract_ado_trajectories(ado_trajectories)
    for t in range(t_horizon):
        fig, ax = plt.subplots()
        plot_scene(ax, t, ado_trajectories, ado_colors, ado_ids, ego_trajectory, ado_trajectories_wo, t_preview, axes)
        plt.savefig(os.path.join(file_path, f"{t:04d}.png"))
        plt.close()


def picture_trajectories(
    file_path: str,
    trajectories: List[torch.Tensor],
    tags: List[str],
    axes: Tuple[Tuple[float, float], Tuple[float, float]] = (
        mantrap.constants.sim_x_axis_default,
        mantrap.constants.sim_y_axis_default,
    ),
):
    """Picture n 2D trajectories in same plot with size determined by the axes. Describe them using given tags."""
    assert len(trajectories) == len(tags), "un-matching number of trajectories and tags"
    fig, ax = plt.subplots()
    for trajectory, tag in zip(trajectories, tags):
        plt.plot(trajectory[:, 0], trajectory[:, 1], label=tag)
    _add_axis_and_grid(axes, ax=ax)
    plt.legend()
    plt.savefig(file_path)
    plt.close()


############################################################################
# Atomic plotting functions  ###############################################
############################################################################


def plot_scene(
    ax: plt.Axes,
    t: int,
    ado_trajectories: torch.Tensor,
    ado_colors: List[List[float]],
    ado_ids: List[str],
    ego_trajectory: torch.Tensor = None,
    ado_trajectories_wo: torch.Tensor = None,
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
    assert check_trajectories(ado_trajectories)
    num_ados, num_modes, t_horizon = extract_ado_trajectories(ado_trajectories)
    assert len(ado_colors) == num_ados, "ado colors must be consistent with trajectories"
    ado_ids = [None] * num_ados if ado_ids is None else ado_ids
    assert len(ado_ids) == num_ados, "ado ids must be consistent with trajectories"
    if ego_trajectory is not None:
        assert check_ego_trajectory(ego_trajectory=ego_trajectory, t_horizon=t_horizon)
    if ado_trajectories_wo is not None:
        assert check_trajectories(ado_trajectories_wo, modes=1, ados=num_ados, t_horizon=t_horizon)

    # Clip position related values to the visualized range, in order to prevent agent representation of the graph.
    ado_trajectories[:, :, :, 0] = torch.clamp(ado_trajectories[:, :, :, 0], min=axes[0][0], max=axes[0][1])
    ado_trajectories[:, :, :, 1] = torch.clamp(ado_trajectories[:, :, :, 1], min=axes[1][0], max=axes[1][1])

    # Plot ados.
    ado_preview = min(preview_horizon, t_horizon - t)
    for ado_i in range(num_ados):
        ado_color = ado_colors[ado_i]

        # Drawing.
        ax = _add_history(ado_trajectories[ado_i, 0, :t, 0:2], color=ado_color, ax=ax)
        for mode_i in range(num_modes):
            ado_id = ado_ids[ado_i] + "_" + str(mode_i)
            ax = _add_agent_representation(ado_trajectories[ado_i, mode_i, t, :], color=ado_color, name=ado_id, ax=ax)
            ax = _add_trajectory(ado_trajectories[ado_i, mode_i, t : t + ado_preview, 0:2], color=ado_color, ax=ax)
        if ado_trajectories_wo is not None:
            ax = _add_wo_trajectory(ado_trajectories_wo[ado_i, 0, :t, 0:2], color=ado_color, ax=ax)

    # Plot ego.
    if ego_trajectory is not None:
        ego_state = ego_trajectory[t, :]
        ego_color = [0, 0, 1.0]
        ego_history = ego_trajectory[:t, 0:2]
        ego_preview = min(preview_horizon, ego_trajectory.shape[0] - t)
        ego_planned = ego_trajectory[t : t + ego_preview, 0:2]

        ax = _add_agent_representation(ego_state, color=ego_color, name="ego", ax=ax)
        ax = _add_history(ego_history, color=ego_color, ax=ax)
        ax = _add_trajectory(ego_planned, color=ego_color, ax=ax)

    # Plot labels, limits and grid.
    ax = _add_axis_and_grid(axes, ax=ax)
    return ax


def _add_agent_representation(state: torch.Tensor, color: List[float], name: Union[str, None], ax: plt.Axes):
    assert check_state(state), "state must be of size 5 or 6 (x, y, theta, vx, vy, t)"
    arrow_length = torch.norm(state[3:5]) / mantrap.constants.agent_speed_max * 0.5

    # Add circle for agent itself.
    ado_circle = plt.Circle(tuple(state[0:2]), mantrap.constants.visualization_agent_radius, color=color, clip_on=True)
    ax.add_artist(ado_circle)

    # Add agent id description.
    if id is not None:
        ax.text(state[0], state[1], name, fontsize=8)

    # Add arrow for orientation and speed.
    rot = torch.tensor([[torch.cos(state[2]), -torch.sin(state[2])], [torch.sin(state[2]), torch.cos(state[2])]])
    darrow = rot.matmul(torch.tensor([1.0, 0.0])) * arrow_length
    head_width = max(0.02, arrow_length / 10)
    plt.arrow(state[0], state[1], darrow[0], darrow[1], head_width=head_width, head_length=0.1, fc="k", ec="k")
    return ax


def _add_trajectory(trajectory: torch.Tensor, color: List[float], ax: plt.Axes):
    assert len(trajectory.shape) == 2, "trajectory must have shape (N, state_length)"
    ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linestyle="-", linewidth=0.6, alpha=1.0)
    return ax


def _add_wo_trajectory(trajectory: torch.Tensor, color: List[float], ax: plt.Axes):
    assert len(trajectory.shape) == 2, "trajectory must have shape (N, state_length)"
    ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linestyle=":", linewidth=0.6, alpha=1.0)
    return ax


def _add_history(history: torch.Tensor, color: List[float], ax: plt.Axes):
    assert len(history.shape) == 2, "history must have shape (M, state_length)"
    ax.plot(history[:, 0], history[:, 1], color=color, linestyle="-.", linewidth=0.6, alpha=0.8)
    return ax


def _add_axis_and_grid(axes: Tuple[Tuple[float, float], Tuple[float, float]], ax: plt.Axes):
    x_axis, y_axis = axes
    ax.set_xlim(xmin=x_axis[0], xmax=x_axis[1])
    ax.set_ylim(ymin=y_axis[0], ymax=y_axis[1])
    ax.tick_params(axis="both", which="major", labelsize=5)
    ax.grid(which="minor", alpha=0.1)
    ax.grid(which="major", alpha=0.3)
    return ax
