import matplotlib.pyplot as plt
import torch

import mantrap.constants
import mantrap.environment

from .atomics import draw_trajectory_axis, draw_samples, draw_agent_representation, interactive_save_image


def visualize_prediction(
    env: mantrap.environment.base.GraphBasedEnvironment,
    ego_planned: torch.Tensor = None,
    ado_planned: torch.Tensor = None,
    ado_planned_wo: torch.Tensor = None,
    ego_goal: torch.Tensor = None,
    legend: bool = False,
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
   :param ado_planned: according ado trajectory conditioned on ego_planned (num_ados, num_samples, t_horizon + 1, 1, 5).
   :param ego_goal: optimization robot goal state.
   :param legend: draw legend in paths plot (might be a mess for many agents).
   :param file_path: storage path, if None return as HTML video object.
   :param ax: optionally the plot can be drawn in an already existing axis.
    """
    # Check inputs for validity in shape and matching sizes.
    if ego_planned is not None:
        assert mantrap.utility.shaping.check_ego_trajectory(ego_planned)
    if ado_planned is not None:
        assert mantrap.utility.shaping.check_ado_samples(ado_planned, ados=env.num_ados)
    if ado_planned_wo is not None:
        assert mantrap.utility.shaping.check_ado_samples(ado_planned_wo, ados=env.num_ados)
    if ego_goal is not None:
        assert mantrap.utility.shaping.check_goal(ego_goal)

    # Create basic plot. In order to safe computational effort the created plot is re-used over the full output
    # video, by deleting the previous frame content and overwrite it (within the `update()` function).
    if ax is None:
        plt.close('all')
        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    # Check ado trajectories and decide whether to mainly plot the without trajectories or the conditioned
    # trajectories (if they are defined).
    ado_trajectory_main = None
    if ado_planned_wo is not None:
        ado_trajectory_main = ado_planned_wo.detach().clone()
    if ado_planned is not None:
        ado_trajectory_main = ado_planned.detach().clone()

    # Plot ego trajectory.
    if ego_planned is not None:
        ego_color, ego_id = env.ego.color, env.ego.id
        ax.plot(ego_planned[:, 0], ego_planned[:, 1], "-", color=ego_color, label=ego_id)
        draw_agent_representation(ego_planned[0, 0:2], color=ego_color, name=ego_id, env_axes=env.axes, ax=ax)

    # Plot ego goal.
    if ego_goal is not None:
        assert mantrap.utility.shaping.check_goal(ego_goal)
        ax.plot(ego_goal[0], ego_goal[1], "rx", markersize=15.0, label="goal")

    # Plot ado trajectories (forward and backward). Additionally plot the ado "side" trajectories, i.e.
    # the unconditioned prediction samples if both samples are defined.
    if ado_trajectory_main is not None:
        for m_ado, ado in enumerate(env.ados):
            ado_trajectory = ado_trajectory_main[m_ado]
            ado_id, ado_color = ado.id, ado.color
            ado_history = ado.history
            ado_position = ado_trajectory[0, 0, 0, 0:2]

            draw_agent_representation(ado_position, name=ado_id, color=ado_color, env_axes=env.axes, ax=ax)
            draw_samples(ado_trajectory, name=ado_id, color=ado_color, ax=ax)
            ax.plot(ado_history[:-1, 0], ado_history[:-1, 1], "-.", color=ado_color, label=ado_id)

            if ado_planned is not None and ado_planned_wo is not None:
                draw_samples(ado_planned_wo[m_ado], name=f"{ado_id}_wo", color=ado_color, ax=ax, alpha=0.5)

    draw_trajectory_axis(env.axes, ax=ax, legend=legend)
    ax.set_title(f"predictions")
    return interactive_save_image(ax, file_path=file_path)
