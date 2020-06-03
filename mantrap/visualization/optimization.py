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
