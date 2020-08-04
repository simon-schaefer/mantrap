import typing

import matplotlib.pyplot as plt
import torch

import mantrap.constants
import mantrap.environment

from .atomics import draw_trajectory, draw_trajectory_axis, draw_samples, draw_agent, interactive_save_image


def visualize_prediction(
    env: mantrap.environment.base.GraphBasedEnvironment,
    ego_planned: torch.Tensor = None,
    ado_planned: torch.Tensor = None,
    ado_planned_wo: torch.Tensor = None,
    ado_actual: torch.Tensor = None,
    ado_histories: torch.Tensor = None,
    ego_goal: torch.Tensor = None,
    display_wo: bool = True,
    legend: bool = False,
    grid: bool = True,
    figsize: typing.Tuple[int, int] = (8, 8),
    title: str = None,
    file_path: str = None,
    ax: plt.Axes = None
):
    """Visualize robot/ado trajectories and distributions.

    Draw trajectories in (quadratic) plot, thereby differentiate the actual trajectories from the
    baseline trajectories (without) using a different line markup. Also add little dots (`plt.Circle`) at the
    agents current positions.

   :param ado_planned_wo: ado trajectories without robot (num_ados, num_samples, t_horizon + 1, num_modes = 1,  5).
   :param env: simulation environment, just used statically here (e.g. to convert ids to agents, roll out
               trajectories, etc.).
   :param ego_planned: planned/optimized ego trajectory (t_horizon + 1, 5).
   :param ado_actual: actual ado trajectories (num_ados, t_horizon + 1, 1, 5).
   :param ado_planned: according ado trajectory conditioned on ego_planned (num_ados, num_samples, t_horizon + 1, 1, 5).
   :param ado_histories: ado history trajectory used instead of internally stored on (num_ados, -1, >=2).
   :param ego_goal: optimization robot goal state.
   :param display_wo: display ado-wo-trajectories.
   :param legend: draw legend in paths plot (might be a mess for many agents).
   :param grid: draw grid background (default = True).
   :param figsize: figure size (default = quadratic (8, 8)), only used if no `ax` is given.
   :param title: plot title (none by default).
   :param file_path: storage path, if None return as HTML video object.
   :param ax: optionally the plot can be drawn in an already existing axis.
    """
    # Check inputs for validity in shape and matching sizes.
    if ego_planned is not None:
        assert mantrap.utility.shaping.check_ego_trajectory(ego_planned)
    if ado_actual is not None:
        assert mantrap.utility.shaping.check_ado_trajectories(ado_actual, ados=env.num_ados, num_modes=1)
    if ado_planned is not None:
        assert mantrap.utility.shaping.check_ado_samples(ado_planned, ados=env.num_ados)
    if ado_planned_wo is not None:
        assert mantrap.utility.shaping.check_ado_samples(ado_planned_wo, ados=env.num_ados)
    if ado_histories is not None:
        assert mantrap.utility.shaping.check_ado_history(ado_histories, ados=env.num_ados)
    if ego_goal is not None:
        assert mantrap.utility.shaping.check_goal(ego_goal)

    # Create basic plot. In order to safe computational effort the created plot is re-used over the full output
    # video, by deleting the previous frame content and overwrite it (within the `update()` function).
    if ax is None:
        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Plot ego trajectory.
    if ego_planned is not None:
        ego_color, ego_id = env.ego.color, env.ego.id
        draw_trajectory(ego_planned, is_robot=True, name=ego_id, color=ego_color, env_axes=env.axes, ax=ax)
        draw_agent(ego_planned[0, 0:4], is_robot=True, color=ego_color, env_axes=env.axes, ax=ax)

    # Plot optimization goal state as a red cross.
    if ego_goal is not None:
        assert mantrap.utility.shaping.check_goal(ego_goal)
        ax.plot(ego_goal[0], ego_goal[1], "rx", markersize=15.0, label="goal")

    # Plot ado trajectories (forward and backward). Additionally plot the ado "side" trajectories, i.e.
    # the unconditioned prediction samples if both samples are defined.
    for m_ado, ado in enumerate(env.ados):
        ado_id, ado_color = ado.id, ado.color

        # History - Ado internal or input ado history.
        ado_history = ado.history if ado_histories is None else ado_histories[m_ado, :, 0:2]
        if len(ado_history) > 1:
            ax.plot(ado_history[:, 0], ado_history[:, 1], "-.", color=ado_color, label=f"{ado_id}_hist", alpha=0.8)
        draw_agent(ado_history[-1, :], is_robot=False, color=ado_color, env_axes=env.axes, ax=ax)

        # Actual trajectory - Miniature pedestrians.
        if ado_actual is not None:
            trajectory = ado_actual[m_ado, :, 0, :]
            draw_trajectory(trajectory, name=ado_id, color=ado_color, env_axes=env.axes, ax=ax, is_robot=False)

        # Sample trajectory - Conditioned and unconditioned.
        if ado_planned is not None:
            draw_samples(ado_planned[m_ado], name=ado_id, color=ado_color, ax=ax, alpha=0.3, marker="-")
        if ado_planned_wo is not None and display_wo:
            draw_samples(ado_planned_wo[m_ado], name=f"{ado_id}_wo", color=ado_color, ax=ax, alpha=0.3, marker=":")

    draw_trajectory_axis(env.axes, ax=ax, legend=legend, grid=grid)
    if title is not None:
        ax.set_title(title)
    return interactive_save_image(ax, file_path=file_path)
