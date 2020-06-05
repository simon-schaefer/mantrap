import os
import typing

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import torch

import mantrap.utility.shaping


def draw_agent_representation(
    position: torch.Tensor,
    name: typing.Union[str, None],
    color: typing.Union[np.ndarray, str],
    env_axes: typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]],
    ax: plt.Axes,
    alpha: float = 1.0
):
    """Add circle for agent and agent id description. If the state (position) is outside of the scene, just
    do not plot it, return directly instead."""
    if not (env_axes[0][0] < position[0] < env_axes[0][1]) or not (env_axes[1][0] < position[1] < env_axes[1][1]):
        return
    position = position.detach().numpy()
    ado_circle = plt.Circle(position, radius=0.2, color=color, clip_on=True, alpha=alpha)
    ax.add_artist(ado_circle)
    if name is not None:
        ax.text(position[0], position[1], name, fontsize=8)
    return ax


def draw_samples(
    samples: torch.Tensor,
    name: typing.Union[str, None],
    color: typing.Union[np.ndarray, str],
    ax: plt.Axes,
    alpha: float = 1.0,
    marker="--"
):
    """Draw trajectory samples into axes, by sample-wise iterations (samples, t_horizon, 1, dims >= 2)."""
    num_samples = samples.shape[0]
    for i in range(num_samples):
        xs, ys = samples[i, :, 0, 0], samples[i, :, 0, 1]
        ax.plot(xs, ys, marker, color=color, label=name, alpha=alpha)
    return ax


def draw_trajectory_axis(env_axes: typing.Tuple[typing.Tuple, typing.Tuple], ax: plt.Axes, legend: bool = True):
    # Set axes limitations for x- and y-axis and add legend and grid to visualization.
    ax.set_xlim(*env_axes[0])
    ax.set_ylim(*env_axes[1])
    ax.grid()
    if legend:
        ax.legend()
    return ax


def output_format(name: str) -> typing.Union[str, None]:
    from mantrap.utility.io import build_os_path, is_running_from_ipython
    interactive = is_running_from_ipython()
    if not interactive:
        output_path = build_os_path(mantrap.constants.VISUALIZATION_DIRECTORY, make_dir=True, free=False)
        output_path = os.path.join(output_path, name)
    else:
        output_path = None
    return output_path


def interactive_save_video(anim: matplotlib.animation.FuncAnimation, file_path: str):
    """In interactive mode (when file_path is not set), return the video itself, otherwise save
    the video in the given directory as a ".gif"-file. """
    if file_path is not None:
        anim.save(f"{file_path}.gif", dpi=60, writer='imagemagick')
    return True if file_path is not None else anim.to_html5_video()


def interactive_save_image(ax: plt.Axes, file_path: str):
    """In interactive mode (when file_path is not set) return the image itself, otherwise save
    the image in the given direction as a ".png"-file. """
    if file_path is not None:
        plt.savefig(f"{file_path}.png")
    return ax
