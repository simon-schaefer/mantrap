import typing

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import torch

import mantrap.constants
import mantrap.environment

from .atomics import __draw_trajectories, __draw_values, __interactive_save_video


def visualize_overview(
    ado_planned_wo: torch.Tensor,
    env: mantrap.environment.base.GraphBasedEnvironment,
    ego_planned: torch.Tensor = None,
    ado_planned: torch.Tensor = None,
    ego_goal: torch.Tensor = None,
    obj_dict: typing.Dict[str, typing.List[torch.Tensor]] = None,
    inf_dict: typing.Dict[str, typing.List[torch.Tensor]] = None,
    ego_trials: typing.List[typing.List[torch.Tensor]] = None,
    plot_path_only: bool = False,
    legend: bool = False,
    kde: bool = False,
    frame_interval: float = mantrap.constants.VISUALIZATION_FRAME_DELAY,
    restart_delay: float = mantrap.constants.VISUALIZATION_RESTART_DELAY,
    file_path: str = None,
):
    """Visualize robot/ado trajectories extended by state or optimization data.

   :param ado_planned_wo: ado trajectories without robot (num_ados, num_modes, t_horizon + 1, 5).
   :param env: simulation environment, just used statically here (e.g. to convert ids to agents, roll out
               trajectories, etc.
   :param ego_planned: planned/optimized ego trajectory (t_horizon + 1, 5).
   :param ado_planned: according ado trajectory conditioned on ego_planned (num_ados, num_modes, t_horizon + 1, 5).
   :param ego_goal: optimization robot goal state.
   :param obj_dict: dictionary storing the objective values by objective module and optimization iteration.
   :param inf_dict: dictionary storing the constraint infeasibility values by constraint module and opt. iteration.
   :param ego_trials: trial ego trajectories during optimization.
   :param plot_path_only: plot only path (main) plot not stats plots.
   :param legend: draw legend in paths plot (might be a mess for many agents).
   :param kde: plot probabilistic trajectory distributions as kde plot (instead of single modes) per timestep.
   :param frame_interval: delay from one to another frame in [ms].
   :param restart_delay: delay for GIF restart after it has been finished [ms].
   :param file_path: storage path, if None return as HTML video object.
    """
    num_vertical_plots = 2  # paths (2)
    num_env_steps = ado_planned_wo.shape[0]
    assert num_env_steps > 1

    # Check inputs for validity in shape and matching sizes.
    if ego_planned is not None:
        assert len(ego_planned.shape) == 3 and ego_planned.shape[0] == num_env_steps
    if ado_planned is not None:
        assert len(ado_planned.shape) == 5 and ado_planned.shape[0] == num_env_steps
    if ego_goal is not None:
        assert mantrap.utility.shaping.check_goal(ego_goal)
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
            ego_goal=ego_goal,
            ego_traj_trials=ego_trials_k,
            legend=legend,
            kde=kde,
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

    anim = matplotlib.animation.FuncAnimation(fig, update, frames=num_env_steps,
                                              interval=frame_interval, repeat_delay=restart_delay)
    return __interactive_save_video(anim, file_path=file_path)
