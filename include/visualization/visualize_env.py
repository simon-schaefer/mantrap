from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_environment(
    ados: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ego: Union[Tuple[np.ndarray, np.ndarray, np.ndarray], None],
    xaxis: Tuple[float, float],
    yaxis: Tuple[float, float],
):
    """Visualize environment using matplotlib.pyplot library. The environment is two-dimensional, has bounds
    `xaxis` and `yaxis` (min, max). It contains agents, which can be seperated in the ados ("obstacles") and
    the ego. Both have some current 2D pose (x, y, theta), history (previous poses) and future trajectories
    (next poses), both history and future stamped with timestamps as a 4D vector (x, y, theta, t). The future
    trajectories of the ados thereby is uncertain, so there could be multiple trajectories given.

    @param ados: list of ados, each having a current pose, history and (multiple) future trajectories.
    @param ego: ego having current pose, history and (one) future trajectory.
    @param xaxis: environment expansion in x direction (min, max).
    @param yaxis: environment expansion in y direction (min, max).
    """
    
