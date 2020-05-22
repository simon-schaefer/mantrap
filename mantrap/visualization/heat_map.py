import typing

import matplotlib.animation
import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

import mantrap.constants

from .atomics import __interactive_save_video


def visualize_heat_map(
    images: np.ndarray,
    bounds: typing.Tuple[typing.List, typing.List],
    color_bounds: typing.Tuple[float, float],
    choices: np.ndarray = None,
    resolution: float = 0.1,
    file_path: str = None,
):
    """Visualize the given images as heat-maps.

    The images are plotted as an animation (GIF) and labeled using the boundaries defined in `bounds`.
    Additionally, e.g. for the visualization of objective and constraint violation values, the "choices"
    made can be added to the plot, e.g. if the images are objective values in dependence on some optimization
    variable, then the variables chosen by the optimizer can be visualized.

    :param images: heat-map images to plot (#steps, N, M).
    :param bounds: lower and upper bounds for image dimensions.
    :param color_bounds: min and max value for heat-map color-bar.
    :param choices: chosen variables as described above (#steps, 2).
    :param resolution: image resolution to convert discrete image grid points to bounds.
    :param file_path: gif storage path (if None, then output is HTML5 video !).
    """
    # Derive image ticks from bounds and resolution data.
    lower, upper = bounds
    assert len(lower) == len(upper) == 2  # 2D (!)
    num_grid_points_x = int((upper[0] - lower[0]) / resolution)
    num_grid_points_y = int((upper[1] - lower[1]) / resolution)
    plot_z_values = choices is not None

    assert len(images.shape) == 3
    if plot_z_values:
        assert len(choices.shape) == 2
        assert images.shape[0] == choices.shape[0]
        assert choices.shape[1] == 2

    # Plot resulting objective value and constraints plot.
    fig, ax = plt.subplots(figsize=(8, 8))
    num_ticks = 8

    # Color map and image definition. However due to a shift in the range of the image data
    # (min, max) and therefore in the colormap, the image has to be re-drawn in every step.
    color_map = matplotlib.cm.get_cmap()
    color_map.set_bad(color="black")

    # Line plot definition, which is also updated during iteration.
    z_values_coords, line = None, None
    if plot_z_values:
        z_values_coords = (choices - np.array(lower)) / resolution
        line, = ax.plot(z_values_coords[0, 0], z_values_coords[0, 1], 'rx')

    def update(k):
        # Reset plot to be re-built from scratch, due to the limitations explained above.
        fig.clear()
        ax = fig.add_subplot(111)

        # Draw heat-map and according color-bar.
        im = ax.imshow(images[k, :, :], interpolation="none", animated=True, cmap=color_map,
                       vmin=color_bounds[0], vmax=color_bounds[1])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', ax=ax)

        # Plot the optimized z-value (if given).
        if plot_z_values:
            line.set_xdata(z_values_coords[k, 0])
            line.set_ydata(z_values_coords[k, 1])
        ax.set_title(f"optimization landscape - step {k}")
        ax.set_xticks(np.linspace(0, num_grid_points_x, num=num_ticks))
        ax.set_xticklabels(np.round(np.linspace(lower[0], upper[0], num=num_ticks), 1))
        ax.set_yticks(np.linspace(0, num_grid_points_y, num=num_ticks))
        ax.set_yticklabels(np.round(np.linspace(lower[1], upper[1], num=num_ticks), 1))
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        return ax

    # Start matplotlib animation with an image per time-step.
    anim = matplotlib.animation.FuncAnimation(fig, update, frames=images.shape[0],
                                              interval=mantrap.constants.VISUALIZATION_FRAME_DELAY,
                                              repeat_delay=mantrap.constants.VISUALIZATION_RESTART_DELAY)
    return __interactive_save_video(anim, file_path=file_path)
