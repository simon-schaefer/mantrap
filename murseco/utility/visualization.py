from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

import murseco.utility.stats


def plot_pdf2d(
    fig: plt.Figure,
    ax: plt.Axes,
    distribution: murseco.utility.stats.Distribution2D,
    xaxis: Tuple[float, float],
    yaxis: Union[Tuple[float, float], None] = None,
    num_points: int = 1e3,
):
    """Plot 2D PDF (probability density function) in mesh grid with the given axes.
    In order to plot the PDF the axes are split up in a constant number of points, i.e. the resolution varies with
    the expansion of the axes. However the default number of points is large enough to draw a fairly accurate
    representation of the distribution while being efficient. If just one axis is given, the other axis is assumed
    to be the same as the given axis.

    :argument fig: matplotlib figure to draw in.
    :argument ax: matplotlib axis to draw in.
    :argument distribution: 2D distribution as subclass of Distribution2D class in utility.stats.
    :argument xaxis: range of x axis as (lower bound, upper bound).
    :argument yaxis: range of y axis as (lower bound, upper bound), assumed to be equal to xaxis if not stated.
    :argument num_points: number of resolution points for axis sampling.
    """
    yaxis = xaxis if yaxis is None else yaxis
    num_points = int(num_points)
    x, y = np.meshgrid(np.linspace(xaxis[0], xaxis[1], num_points), np.linspace(yaxis[0], yaxis[1], num_points))
    pdf = distribution.pdf_at(x, y)
    color_mesh = ax.pcolormesh(x, y, pdf, cmap='gist_earth')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(color_mesh, ax=ax)
