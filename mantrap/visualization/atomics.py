import os
import typing

import matplotlib.animation
import matplotlib.image
import matplotlib.offsetbox
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions

import mantrap.constants
import mantrap.utility.io
import mantrap.utility.shaping


def draw_agent(state: typing.Union[torch.Tensor, np.ndarray], color: typing.Union[np.ndarray, str],
               env_axes: typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]],
               ax: plt.Axes, is_robot: bool = False, scale: float = 1.0):
    """Add circle for agent and agent id description. If the state (position) is outside of the scene, just
    do not plot it, return directly instead (state = position or position+velocity)."""
    if not (env_axes[0][0] < state[0] < env_axes[0][1]) or not (env_axes[1][0] < state[1] < env_axes[1][1]):
        return
    if type(state) is torch.Tensor:
        state = state.detach().numpy()

    # Read image (differentiate between robot and pedestrian).
    if is_robot:
        image_path = mantrap.utility.io.build_os_path(os.path.join("third_party", "visualization", "robot.png"))
    else:
        image_path = mantrap.utility.io.build_os_path(os.path.join("third_party", "visualization", "walking.png"))
    image = matplotlib.image.imread(image_path)

    # Rotate image in correct orientation (based on velocity vector if given inside state).
    black_index = np.where(np.isclose(image[:, :, 0:3], np.zeros(3), atol=0.1))
    image[black_index[0], black_index[1], 0:3] = color
    if state.size > 2:
        orientation = np.arctan2(state[3], state[2])  # theta = arctan2(vy/vx)
        if -np.pi/2 < orientation < np.pi/2:
            image = np.fliplr(image)

    # Transparent white background.
    color_norm = np.linalg.norm(image, axis=2)
    for ix in range(image.shape[0]):
        for iy in range(image.shape[1]):
            color_norm = np.sum(image[ix, iy, :3])
            image[ix, iy, -1] = 1.0 if color_norm < 2.0 else 0.0

    # Draw image in axes using `OffsetImage`.
    image_box = matplotlib.offsetbox.OffsetImage(image, zoom=scale)
    ab = matplotlib.offsetbox.AnnotationBbox(image_box, (state[0], state[1]), frameon=False)
    ax.add_artist(ab)

    return ax


def draw_gmm(distribution: torch.distributions.Distribution,
             color: typing.Union[np.ndarray, str], ax: plt.Axes, alpha: float = 1.0, num_modes: int = 5):
    """Draw positional distribution (GMM) as elliptic shapes, i.e. draw each Gaussian of the GMM as an
    ellipse with the center as its mean and axes as its standard deviation. """
    mean, var = distribution.mean, distribution.stddev
    t_horizon, dist_modes, _ = mean.shape
    if dist_modes < num_modes:
        num_modes = dist_modes

    for t in range(t_horizon):
        for n_mode in range(num_modes):
            xy = tuple(mean[t, n_mode, :].tolist())
            wh = var[t, n_mode, :].tolist()
            ellipse = matplotlib.patches.Ellipse(xy, width=wh[0], height=wh[1], alpha=alpha, color=color)
            ax.add_patch(ellipse)

    return ax


def draw_samples(samples: torch.Tensor, name: typing.Union[str, None], color: typing.Union[np.ndarray, str],
                 ax: plt.Axes, marker: str = "--", alpha: float = 1.0):
    """Draw trajectory samples into axes, by sample-wise iterations (samples, t_horizon, 1, dims >= 2)."""
    num_samples = samples.shape[0]
    for i in range(num_samples):
        xs, ys = samples[i, :, 0, 0], samples[i, :, 0, 1]
        kwargs = {"label": name} if i == 0 else {}
        ax.plot(xs, ys, marker, color=color, alpha=alpha, **kwargs)
        # ax.plot(xs, ys, "o", color=color, alpha=alpha)  # time-step marker
    return ax


def draw_trajectory(trajectory: torch.Tensor, name: typing.Union[str, None], color: typing.Union[np.ndarray, str],
                    env_axes: typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]],
                    ax: plt.Axes, marker: str = "-", is_robot: bool = False):
    """Draw trajectory with miniature agent representations on the path."""
    trajectory = trajectory.detach().numpy()
    ax.plot(trajectory[:, 0], trajectory[:, 1], marker, color=color, label=name)
    # ax.plot(trajectory[:, 0], trajectory[:, 1], "o", color=color)
    for t in range(2, trajectory.shape[0], 2):  # every second element
        draw_agent(trajectory[t, :], color=color, env_axes=env_axes, ax=ax, is_robot=is_robot, scale=0.5)
    return ax


def draw_trajectory_axis(env_axes: typing.Tuple[typing.Tuple, typing.Tuple],
                         ax: plt.Axes, legend: bool = True, grid: bool = True):
    # Set axes limitations for x- and y-axis and add legend and grid to visualization.
    ax.set_xlim(*env_axes[0])
    ax.set_ylim(*env_axes[1])
    if grid:
        ax.grid()
    if legend:
        ax.legend()
    return ax


def output_format(name: str, force_save: bool = False) -> typing.Union[str, None]:
    from mantrap.utility.io import build_os_path, is_running_from_ipython
    interactive = is_running_from_ipython()
    if not interactive or force_save:
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
