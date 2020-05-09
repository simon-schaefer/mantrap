import typing

import matplotlib.animation
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import torch

import mantrap.environment
import mantrap.utility.maths
import mantrap.utility.shaping


def visualize(
    ado_planned_wo: torch.Tensor,
    env: mantrap.environment.base.GraphBasedEnvironment,
    ego_planned: torch.Tensor = None,
    ado_planned: torch.Tensor = None,
    file_path: str = None,
    obj_dict: typing.Dict[str, typing.List[torch.Tensor]] = None,
    inf_dict: typing.Dict[str, typing.List[torch.Tensor]] = None,
    ego_trials: typing.List[typing.List[torch.Tensor]] = None,
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
        axs[0] = __draw_trajectories(
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

            assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, pos_and_vel_only=True)
            assert mantrap.utility.shaping.check_ado_trajectories(ado_trajectories, t_horizon, ados=env.num_ados)

            time_axis = ego_trajectory[:, -1].detach().numpy()

            # Determine velocity norm (L2) at every point of time for ado and ego trajectories.
            ado_velocity_norm = np.linalg.norm(ado_trajectories[:, :, :, 2:4].detach().numpy(), axis=3)
            # Determine acceleration norm (L2) at every point of time for ado and ego trajectories.
            dd = mantrap.utility.maths.Derivative2(horizon=t_horizon, dt=env.dt, velocity=True)
            ado_acceleration_norm = np.linalg.norm(dd.compute(ado_trajectories[:, :, :, 2:4]).detach().numpy(), axis=3)
            # Determine ego controls norm (L2) at every point of time for ado and ego trajectories.
            ego_controls = env.ego.roll_trajectory(trajectory=ego_trajectory, dt=env.dt)
            ego_controls_norm = np.linalg.norm(ego_controls.detach().numpy(),  axis=1)
            ego_controls_norm = np.concatenate((ego_controls_norm, np.zeros(1)))  # controls = T - 1 so stretch them

            # Draw resulting ado agent states in the different axes plots.
            for ghost in env.ghosts:
                i_ado, i_mode = env.convert_ghost_id(ghost_id=ghost.id)
                ado_kwargs = {"color": ghost.agent.color, "label": f"{ghost.id}_current"}
                __draw_values(ado_velocity_norm[i_ado, i_mode, :], time_axis, ax=axs[1], k=k, **ado_kwargs)
                __draw_values(ado_acceleration_norm[i_ado, i_mode, :], time_axis, ax=axs[2], k=k, **ado_kwargs)
            axs[1].set_title("velocities [m/s]")
            axs[2].set_title("accelerations [m/s^2]")
            for i in [1, 2]:
                axs[i].grid()
                axs[i].legend()

            # Draw ego control input in separate plot, together with its lower and upper bound.
            ego_kwargs = {"color": env.ego.color, "label": env.ego.id}
            lower, upper = env.ego.control_limits()
            axs[3].plot(time_axis, np.ones(time_axis.size) * lower, "--", label="lower")
            axs[3].plot(time_axis, np.ones(time_axis.size) * upper, "--", label="upper")
            __draw_values(ego_controls_norm, time_axis, ax=axs[3], k=k, **ego_kwargs)
            axs[3].set_title("control input")

            # Draw objective and constraint violation values in line plot.
            if obj_dict is not None and inf_dict is not None:
                def transform_data(raw_data: torch.Tensor) -> np.ndarray:
                    assert len(raw_data.shape) == 1  # one-dimensional vector (!)
                    return np.log(np.asarray(raw_data) + 1e-8)

                for key, values in obj_dict.items():
                    axs[4] = __draw_values(transform_data(values[k]), time_axis, label=key, ax=axs[4], k=k)
                axs[4].set_title("log-objectives")
                for key, values in inf_dict.items():
                    axs[5] = __draw_values(transform_data(values[k]), time_axis, label=key, ax=axs[5], k=k)
                axs[5].set_title("log-constraints")

        axs[0].set_title(f"step {k}")
        return axs

    anim = matplotlib.animation.FuncAnimation(fig, update, frames=num_env_steps - 1, interval=300)

    # In interactive mode (when file_path is not set), return the video itself, otherwise save
    # the video in the given directory as a ".gif"-file.
    if file_path is not None:
        anim.save(f"{file_path}.gif", dpi=60, writer='imagemagick')
    return True if file_path is not None else anim.to_html5_video()


# ##########################################################################
# Optimization Heat Map ####################################################
# ##########################################################################
def visualize_heat_map(
    images: np.ndarray,
    z_bounds: typing.Tuple[typing.List, typing.List],
    z_values: np.ndarray,
    resolution: float = 0.1,
    file_path: str = None,
):
    # Derive image ticks from bounds and resolution data.
    lower, upper = z_bounds
    assert len(lower) == len(upper) == 2  # 2D (!)
    num_grid_points_x = int((upper[0] - lower[0]) / resolution)
    num_grid_points_y = int((upper[1] - lower[1]) / resolution)
    assert len(images.shape) == 3
    assert images.shape[1] == num_grid_points_x
    assert images.shape[2] == num_grid_points_y
    assert len(z_values.shape) == 2
    assert images.shape[0] == z_values.shape[0]
    assert z_values.shape[1] == 2

    # Plot resulting objective value and constraints plot.
    fig, ax = plt.subplots(figsize=(8, 8))
    num_ticks = 8
    ax.set_xticks(np.linspace(0, num_grid_points_x, num=num_ticks))
    ax.set_xticklabels(np.round(np.linspace(lower[0], upper[0], num=num_ticks), 1))
    ax.set_yticks(np.linspace(0, num_grid_points_y, num=num_ticks))
    ax.set_yticklabels(np.round(np.linspace(lower[1], upper[1], num=num_ticks), 1))
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")

    # Color map and image definition. However due to a shift in the range of the image data
    # (min, max) and therefore in the colormap, the image has to be re-drawn in every step.
    color_map = matplotlib.cm.get_cmap()
    color_map.set_bad(color="black")
    im = ax.imshow(images[0, :, :], interpolation="none", animated=True)
    cb = fig.colorbar(im)

    # Line plot definition, which is also updated during iteration.
    z_values_coords = (z_values - np.array(lower)) / resolution
    line, = ax.plot(z_values_coords[0, 0], z_values_coords[0, 1], 'rx')

    def update(k):
        im = ax.imshow(images[k, :, :], interpolation="none", animated=True)
        line.set_xdata(z_values_coords[k, 0])
        line.set_ydata(z_values_coords[k, 1])
        ax.set_title(f"optimization landscape - step {k}")
        return ax

    # Start matplotlib animation with an image per time-step.
    anim = matplotlib.animation.FuncAnimation(fig, update, frames=images.shape[0] - 1, interval=2000, repeat_delay=2000)

    # In interactive mode (when file_path is not set), return the video itself, otherwise save
    # the video in the given directory as a ".gif"-file.
    if file_path is not None:
        anim.save(f"{file_path}.gif", dpi=60, writer='imagemagick')
    return True if file_path is not None else anim.to_html5_video()


# ##########################################################################
# Atomic plotting functions ################################################
# ##########################################################################
def __draw_trajectories(
    ado_trajectories_wo: torch.Tensor,
    env: mantrap.environment.base.GraphBasedEnvironment,
    ax: plt.Axes,
    ego_trajectory: torch.Tensor = None,
    ado_trajectories: torch.Tensor = None,
    ego_traj_trials: typing.List[torch.Tensor] = None
):
    """Plot current and base solution in the scene. This includes the determined ego trajectory (x) as well as the
    resulting ado trajectories based on some environment."""

    def draw_agent_representation(state: torch.Tensor, color: np.ndarray, name: typing.Union[str, None]):
        """Add circle for agent and agent id description. If the state (position) is outside of the scene, just
        do not plot it, return directly instead."""
        assert mantrap.utility.shaping.check_ego_state(state, enforce_temporal=False)

        if not (env.axes[0][0] < state[0] < env.axes[0][1]) or not (env.axes[1][0] < state[1] < env.axes[1][1]):
            return
        state = state.detach().numpy()
        ado_circle = plt.Circle(state[0:2], 0.2, color=color, clip_on=True)
        ax.add_artist(ado_circle)
        if id is not None:
            ax.text(state[0], state[1], name, fontsize=8)

    assert mantrap.utility.shaping.check_ado_trajectories(ado_trajectories_wo, ados=env.num_ados)
    planning_horizon = ado_trajectories_wo.shape[2]

    # Plot ego trajectory.
    if ego_trajectory is not None:
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, planning_horizon, pos_and_vel_only=True)
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
            assert mantrap.utility.shaping.check_ado_trajectories(ado_trajectories, planning_horizon, ados=env.num_ados)
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
