import typing

import matplotlib.animation
import matplotlib.pyplot as plt
import torch

import mantrap.constants
import mantrap.environment
import mantrap.utility.shaping

from .atomics import interactive_save_video
from .prediction import visualize_prediction


def visualize_optimization(
    ego_planned: torch.Tensor,
    ado_planned: typing.List[torch.Tensor],
    ado_planned_wo: typing.List[torch.Tensor],
    env: mantrap.environment.base.GraphBasedEnvironment,
    ego_goal: torch.Tensor = None,
    legend: bool = False,
    frame_interval: float = mantrap.constants.VISUALIZATION_FRAME_DELAY,
    restart_delay: float = mantrap.constants.VISUALIZATION_RESTART_DELAY,
    file_path: str = None,
):
    """Visualize robot/ado trajectories and distributions over full optimization (multiple time-steps).

    Draw trajectories in (quadratic) plot, thereby differentiate the actual trajectories from the
    baseline trajectories (without) using a different line markup. Also add little dots (`plt.Circle`) at the
    agents current positions.

   :param ado_planned_wo: ado trajectories without robot
                          (time-steps, num_ados, num_samples, t_horizon + 1, num_modes = 1,  5).
   :param env: simulation environment, just used statically here (e.g. to convert ids to agents, roll out
               trajectories, etc.).
   :param ego_planned: planned/optimized ego trajectory (time-step, t_horizon + 1, 5).
   :param ado_planned: according ado trajectory conditioned on ego_planned
                       (time-steps, num_ados, num_samples, t_horizon + 1, 1, 5).
   :param ego_goal: optimization robot goal state.
   :param legend: draw legend in paths plot (might be a mess for many agents).
   :param file_path: storage path, if None return as HTML video object.
   :param frame_interval: video frame displaying time interval [ms].
   :param restart_delay: video restart delay time interval [ms].
    """
    time_steps = ego_planned.shape[0]
    assert all(mantrap.utility.shaping.check_ego_trajectory(ego_planned[k]) for k in range(time_steps))
    assert len(ado_planned) == time_steps
    assert len(ado_planned_wo) == time_steps

    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    def update(k):
        plt.axis("off")
        ax.cla()

        visualize_prediction(ego_planned=ego_planned[k], ado_planned=ado_planned[k], ado_planned_wo=ado_planned_wo[k],
                             env=env, ego_goal=ego_goal, legend=legend, ax=ax)
        return ax

    anim = matplotlib.animation.FuncAnimation(fig, update, frames=time_steps,
                                              interval=frame_interval, repeat_delay=restart_delay)
    return interactive_save_video(anim, file_path=file_path)
