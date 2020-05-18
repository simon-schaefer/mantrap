import typing

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import torch

import mantrap.utility.shaping


def __draw_agent_representation(
    state: torch.Tensor,
    name: typing.Union[str, None],
    color: typing.Union[np.ndarray, str],
    env_axes: typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]],
    ax: plt.Axes,
):
    """Add circle for agent and agent id description. If the state (position) is outside of the scene, just
    do not plot it, return directly instead."""
    assert mantrap.utility.shaping.check_ego_state(state, enforce_temporal=False)

    if not (env_axes[0][0] < state[0] < env_axes[0][1]) or not (env_axes[1][0] < state[1] < env_axes[1][1]):
        return
    state = state.detach().numpy()
    ado_circle = plt.Circle(state[0:2], radius=0.2, color=color, clip_on=True)
    ax.add_artist(ado_circle)
    if name is not None:
        ax.text(state[0], state[1], name, fontsize=8)
    return ax


def __draw_trajectory_axis(
    env_axes: typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]],
    ax: plt.Axes
):
    # Set axes limitations for x- and y-axis and add legend and grid to visualization.
    ax.set_xlim(*env_axes[0])
    ax.set_ylim(*env_axes[1])
    ax.grid()
    ax.legend()
    return ax


def __draw_trajectories(
    ado_trajectories_wo: torch.Tensor,
    env: mantrap.environment.base.GraphBasedEnvironment,
    ax: plt.Axes,
    ego_trajectory: torch.Tensor = None,
    ado_trajectories: torch.Tensor = None,
    ego_goal: typing.Union[torch.Tensor, None] = None,
    ego_traj_trials: typing.List[torch.Tensor] = None
):
    """Plot current and base solution in the scene. This includes the determined ego trajectory (x) as well as the
    resulting ado trajectories based on some environment."""
    assert mantrap.utility.shaping.check_ado_trajectories(ado_trajectories_wo, ados=env.num_ados)
    planning_horizon = ado_trajectories_wo.shape[2]

    # Plot ego trajectory.
    if ego_trajectory is not None:
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, planning_horizon, pos_and_vel_only=True)
        ego_trajectory_np = ego_trajectory.detach().numpy()
        ego_color, ego_id = env.ego.color, env.ego.id
        ax.plot(ego_trajectory_np[:, 0], ego_trajectory_np[:, 1], "-", color=ego_color, label=ego_id)
        ax = __draw_agent_representation(ego_trajectory[0, :], color=ego_color, name=ego_id, env_axes=env.axes, ax=ax)

    # Plot ego goal.
    if ego_goal is not None:
        assert mantrap.utility.shaping.check_goal(ego_goal)
        ax.plot(ego_goal[0], ego_goal[1], "rx", markersize=15.0, label="goal")

    # Plot trial trajectories during optimisation process.
    if ego_traj_trials is not None:
        for ego_traj_trial in ego_traj_trials:
            ego_traj_trial_np = ego_traj_trial.detach().numpy()
            ax.plot(ego_traj_trial_np[:, 0], ego_traj_trial_np[:, 1], "--", color=env.ego.color, alpha=0.04)

    # Plot current and base resulting simulated ado trajectories in the scene.
    for ghost in env.ghosts:
        i_ado, i_mode = env.convert_ghost_id(ghost_id=ghost.id)
        ado_id, ado_color = ghost.id, ghost.agent.color
        ado_pos_wo = ado_trajectories_wo[i_ado, i_mode, :, 0:2].detach().numpy()

        ax.plot(ado_pos_wo[:, 0], ado_pos_wo[:, 1], "--", color=ado_color, label=f"{ado_id}_wo")
        if ado_trajectories is not None:
            assert mantrap.utility.shaping.check_ado_trajectories(ado_trajectories, planning_horizon, ados=env.num_ados)
            ado_pos = ado_trajectories[i_ado, i_mode, :, 0:2].detach().numpy()
            ax.plot(ado_pos[:, 0], ado_pos[:, 1], "-*", color=ado_color, label=f"{ado_id}")
            agent_rep_state = ado_trajectories[i_ado, i_mode, 0, :]
        else:
            agent_rep_state = ado_trajectories_wo[i_ado, i_mode, 0, :]
        ax = __draw_agent_representation(agent_rep_state, color=ado_color, name=ado_id, env_axes=env.axes, ax=ax)

    ax = __draw_trajectory_axis(env.axes, ax=ax)
    return ax


def __draw_values(
    values: np.ndarray,
    time_axis: np.ndarray,
    ax: plt.Axes,
    color: np.ndarray = np.zeros(3),
    label: typing.Union[str, None] = None,
    k: typing.Union[int, None] = 0
):
    assert len(values.shape) == 1  # one-dimensional vector (!)

    ax.plot(time_axis, values, label=label, color=color)
    if k is not None:
        ax.axvline(x=time_axis[k], color=np.array([1, 0, 0]))
    ax.legend()
    ax.grid()
    return ax


def __interactive_save_video(anim: matplotlib.animation.FuncAnimation, file_path: str):
    """In interactive mode (when file_path is not set), return the video itself, otherwise save
    the video in the given directory as a ".gif"-file. """
    if file_path is not None:
        anim.save(f"{file_path}.gif", dpi=60, writer='imagemagick')
    return True if file_path is not None else anim.to_html5_video()


def __interactive_save_image(fig: plt.Figure, file_path: str):
    """In interactive mode (when file_path is not set) return the image itself, otherwise save
    the image in the given direction as a ".png"-file. """
    if file_path is not None:
        plt.savefig(f"{file_path}.png")
    return True
