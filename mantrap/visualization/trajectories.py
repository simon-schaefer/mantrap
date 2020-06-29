import typing

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import torch

import mantrap.constants

from .atomics import draw_agent, draw_trajectory_axis, interactive_save_video


def visualize_trajectories(
    trajectories: typing.List[torch.Tensor],
    labels: typing.List[str],
    colors: typing.List[typing.Union[np.ndarray, str]] = None,
    env_axes: typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]] = ((-10, 10), (-10, 10)),
    frame_interval: float = mantrap.constants.VISUALIZATION_FRAME_DELAY,
    restart_delay: float = mantrap.constants.VISUALIZATION_RESTART_DELAY,
    file_path: str = None,
):
    assert len(trajectories[0].shape) == 2
    t_horizon = trajectories[0].shape[0]
    assert all([mantrap.utility.shaping.check_ego_trajectory(x, t_horizon, pos_only=True) for x in trajectories])
    if colors is None:
        colors = [np.array([0, 0, 1])] * len(trajectories)  # blue = default
    assert len(labels) == len(colors) == len(trajectories)

    # Create basic plot. In order to safe computational effort the created plot is re-used over the full output
    # video, by deleting the previous frame content and overwrite it (within the `update()` function).
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.axis("off")

    def update(k):
        plt.axis("off")
        ax.cla()

        # Draw trajectories in (quadratic) plot, thereby differentiate the trajectories by color. Also add
        # little dots (`plt.Circle`) at the agents current positions.
        for label, color, trajectory in zip(labels, colors,  trajectories):
            draw_agent(trajectory[k, :].detach(), color=color, env_axes=env_axes, ax=ax)
            ax.plot(trajectory[:k, 0].detach(), trajectory[:k, 1].detach(), "--", color=color, label=label)

        draw_trajectory_axis(env_axes, ax=ax)
        ax.set_title(f"step {k}")
        return ax

    anim = matplotlib.animation.FuncAnimation(fig, update, frames=t_horizon,
                                              interval=frame_interval, repeat_delay=restart_delay)
    return interactive_save_video(anim, file_path=file_path)
