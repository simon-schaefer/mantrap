from typing import Dict, List, Union

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import torch

from mantrap.environment.environment import GraphBasedEnvironment
from mantrap.utility.maths import Derivative2
from mantrap.utility.shaping import check_ego_trajectory, check_ado_trajectories, check_ego_state


def visualize(
    ado_planned_wo: torch.Tensor,
    env: GraphBasedEnvironment,
    ego_planned: torch.Tensor = None,
    ado_planned: torch.Tensor = None,
    file_path: str = None,
    obj_dict: Dict[str, List[torch.Tensor]] = None,
    inf_dict: Dict[str, List[torch.Tensor]] = None,
    ego_trials: List[List[torch.Tensor]] = None,
    plot_path_only: bool = False
):
    num_vertical_plots = 2  # paths (2)
    num_env_steps = ado_planned_wo.shape[0]
    assert num_env_steps > 1

    # Check inputs for validity in shape and matching sizes.
    if ego_planned is not None:
        assert len(ego_planned.shape) == 3 and ego_planned.shape[0] == num_env_steps

    if ado_planned is not None:
        assert len(ado_planned.shape) == 5 and ado_planned.shape[0] == num_env_steps

    if not plot_path_only:
        assert ego_planned is not None and ado_planned is not None
        num_vertical_plots += 3  # velocities, accelerations, controls (3)
        if obj_dict is not None and inf_dict is not None and not plot_path_only:
            assert all([len(x) == num_env_steps for x in obj_dict.values()])
            assert all([len(x) == num_env_steps for x in inf_dict.values()])
            num_vertical_plots += 2  # objective and constraints (2)

    # Create basic plot. In order to safe computational effort the created plot is re-used over the full output
    # video, by deleting the previous frame content and overwrite it (within the `update()` function).
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    grid = plt.GridSpec(num_vertical_plots, 2, wspace=0.4, hspace=0.3, figure=fig)
    plt.axis("off")

    axs = list()
    # Trajectory plot.
    axs.append(fig.add_subplot(grid[:3, :]))
    # Velocity, acceleration and control plot.
    if not plot_path_only:
        axs.append(fig.add_subplot(grid[2 + 1, :]))
        axs.append(fig.add_subplot(grid[2 + 2, :]))
        axs.append(fig.add_subplot(grid[2 + 3, :]))
        # Objective & Constraint plot.
        if obj_dict is not None and inf_dict is not None:
            axs.append(fig.add_subplot(grid[2 + 4, 0]))
            axs.append(fig.add_subplot(grid[2 + 4, 1]))

    def update(k):

        ego_planned_k = ego_planned[k].detach() if ego_planned is not None else None
        ado_planned_k = ado_planned[k].detach() if ado_planned is not None else None
        ado_planned_wo_k = ado_planned_wo[k].detach()
        ego_trials_k = ego_trials[k] if ego_trials is not None else None

        # Reset plotting axes in order to not having to re-build the whole plot (efficiency !).
        plt.axis("off")
        for i in range(len(axs)):
            axs[i].cla()

        # Draw trajectories in uppermost (quadratic) plot, thereby differentiate the actual trajectories from the
        # baseline trajectories using a different line markup. Also add little dots (`plt.Circle`) at the
        # agents current positions.
        axs[0] = draw_trajectories(
            ego_trajectory=ego_planned_k,
            ado_trajectories=ado_planned_k,
            ado_trajectories_wo=ado_planned_wo_k,
            env=env,
            ego_traj_trials=ego_trials_k,
            ax=axs[0]
        )

        # Draw remaining state graphs, such as velocities, acceleration, controls, optimization performance, etc.
        # As asserted above, when `plot_path_only = False` then `ego_planned != None` (!), so no checks are required
        # within the if-bracket.
        if not plot_path_only:
            ego_trajectory = ego_planned[:, 0, :].detach()
            ado_trajectories = ado_planned[:, :, :, 0, :].permute(1, 2, 0, 3).detach()
            t_horizon = ego_trajectory.shape[0]

            assert check_ego_trajectory(ego_trajectory, pos_and_vel_only=True)
            assert check_ado_trajectories(ado_trajectories, ados=env.num_ados, t_horizon=t_horizon)

            time_axis = ego_trajectory[:, -1].detach().numpy()

            # Determine velocity norm (L2) at every point of time for ado and ego trajectories.
            ado_velocity_norm = np.linalg.norm(ado_trajectories[:, :, :, 2:4].detach().numpy(), axis=3)
            # Determine acceleration norm (L2) at every point of time for ado and ego trajectories.
            dd = Derivative2(horizon=t_horizon, dt=env.dt, velocity=True)
            ado_acceleration_norm = np.linalg.norm(dd.compute(ado_trajectories[:, :, :, 2:4]).detach().numpy(), axis=3)
            # Determine ego controls norm (L2) at every point of time for ado and ego trajectories.
            ego_controls = env.ego.roll_trajectory(trajectory=ego_trajectory, dt=env.dt)
            ego_controls_norm = np.linalg.norm(ego_controls.detach().numpy(),  axis=1)
            ego_controls_norm = np.concatenate((ego_controls_norm, np.zeros(1)))  # controls = T - 1 so stretch them

            # Draw calculation results in the different axes plots.
            for ghost in env.ghosts:
                i_ado, i_mode = env.convert_ghost_id(ghost_id=ghost.id)
                ado_kwargs = {"color": ghost.agent.color, "label": f"{ghost.id}_current"}
                draw_values(ado_velocity_norm[i_ado, i_mode, :], time_axis, ax=axs[1], k=k, **ado_kwargs)
                draw_values(ado_acceleration_norm[i_ado, i_mode, :], time_axis, ax=axs[2], k=k, **ado_kwargs)
            ego_kwargs = {"color": env.ego.color, "label": env.ego.id}
            draw_values(ego_controls_norm, time_axis, ax=axs[3], k=k, **ego_kwargs)
            axs[1].set_title("velocities [m/s]")
            axs[2].set_title("accelerations [m/s^2]")
            axs[3].set_title("control input")
            for i in [1, 2]:
                axs[i].grid()
                axs[i].legend()

            # Draw objective and constraint violation values in line plot.
            if obj_dict is not None and inf_dict is not None:
                def transform_data(raw_data: torch.Tensor) -> np.ndarray:
                    assert len(raw_data.shape) == 1  # one-dimensional vector (!)
                    return np.log(np.asarray(raw_data) + 1e-8)

                for key, values in obj_dict.items():
                    axs[4] = draw_values(transform_data(values[k]), time_axis, label=key, ax=axs[4], k=k)
                for key, values in inf_dict.items():
                    axs[5] = draw_values(transform_data(values[k]), time_axis, label=key, ax=axs[5], k=k)

        axs[0].set_title(f"step {k}")
        return axs

    anim = FuncAnimation(fig, update, frames=num_env_steps - 1, interval=300)

    # In interactive mode (when file_path is not set), return the video itself, otherwise save the video at
    # the given directory as a ".gif"-file.
    if file_path is not None:
        anim.save(f"{file_path}.gif", dpi=60, writer='imagemagick')
    return True if file_path is not None else anim.to_html5_video()


# ##########################################################################
# Atomic plotting functions ################################################
# ##########################################################################
def draw_trajectories(
    ado_trajectories_wo: torch.Tensor,
    env: GraphBasedEnvironment,
    ax: plt.Axes,
    ego_trajectory: torch.Tensor = None,
    ado_trajectories: torch.Tensor = None,
    ego_traj_trials: List[torch.Tensor] = None
):
    """Plot current and base solution in the scene. This includes the determined ego trajectory (x) as well as the
    resulting ado trajectories based on some environment."""

    def draw_agent_representation(state: torch.Tensor, color: np.ndarray, name: Union[str, None]):
        """Add circle for agent and agent id description. If the state (position) is outside of the scene, just
        do not plot it, return directly instead."""
        assert check_ego_state(state, enforce_temporal=False)

        if not (env.axes[0][0] < state[0] < env.axes[0][1]) or not (env.axes[1][0] < state[1] < env.axes[1][1]):
            return
        state = state.detach().numpy()
        ado_circle = plt.Circle(state[0:2], 0.2, color=color, clip_on=True)
        ax.add_artist(ado_circle)
        if id is not None:
            ax.text(state[0], state[1], name, fontsize=8)

    assert check_ado_trajectories(ado_trajectories_wo, ados=env.num_ados)
    planning_horizon = ado_trajectories_wo.shape[2]

    # Plot ego trajectory.
    if ego_trajectory is not None:
        assert check_ego_trajectory(ego_trajectory, pos_and_vel_only=True, t_horizon=planning_horizon)
        ego_trajectory_np = ego_trajectory.detach().numpy()
        ax.plot(ego_trajectory_np[:, 0], ego_trajectory_np[:, 1], "-", color=env.ego.color, label=env.ego.id)
        draw_agent_representation(ego_trajectory[0, :], color=env.ego.color, name=env.ego.id)

    # Plot trial trajectories during optimisation process.
    if ego_traj_trials is not None:
        for ego_traj_trial in ego_traj_trials:
            ego_traj_trial_np = ego_traj_trial.detach().numpy()
            ax.plot(ego_traj_trial_np[:, 0], ego_traj_trial_np[:, 1], "--", color=env.ego.color, alpha=0.08)

    # Plot current and base resulting simulated ado trajectories in the scene.
    for ghost in env.ghosts:
        i_ado, i_mode = env.convert_ghost_id(ghost_id=ghost.id)
        ado_id, ado_color = ghost.id, ghost.agent.color
        ado_pos_wo = ado_trajectories_wo[i_ado, i_mode, :, 0:2].detach().numpy()

        ax.plot(ado_pos_wo[:, 0], ado_pos_wo[:, 1], "--", color=ado_color, label=f"{ado_id}_wo")
        if ado_trajectories is not None:
            assert check_ado_trajectories(ado_trajectories, ados=env.num_ados, t_horizon=planning_horizon)
            ado_pos = ado_trajectories[i_ado, i_mode, :, 0:2].detach().numpy()
            ax.plot(ado_pos[:, 0], ado_pos[:, 1], "-*", color=ado_color, label=f"{ado_id}")
            agent_rep_state = ado_trajectories[i_ado, i_mode, 0, :]
        else:
            agent_rep_state = ado_trajectories_wo[i_ado, i_mode, 0, :]
        draw_agent_representation(agent_rep_state, color=ado_color, name=ado_id)

    # Set axes limitations for x- and y-axis and add legend and grid to visualization.
    ax.set_xlim(*env.axes[0])
    ax.set_ylim(*env.axes[1])
    ax.grid()
    ax.legend()
    return ax


def draw_values(
    values: np.ndarray,
    time_axis: np.ndarray,
    ax: plt.Axes,
    color: np.ndarray = np.zeros(3),
    label: Union[str, None] = None,
    k: Union[int, None] = 0
):
    assert len(values.shape) == 1  # one-dimensional vector (!)

    ax.plot(time_axis, np.log(np.asarray(values) + 1e-8), label=label, color=color)
    if k is not None:
        ax.axvline(x=time_axis[k], color=np.array([1, 0, 0]))
    ax.legend()
    ax.grid()
    return ax
