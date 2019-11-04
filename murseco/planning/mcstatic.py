import logging
from typing import Tuple, Union

import numpy as np

from murseco.problem import DTCSProblem


def monte_carlo_static(
    problem: DTCSProblem
) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None]]:

    logging.debug("starting DTCS monte_carlo_static planning")
    (x_min, x_max), (y_min, y_max) = problem.environment.xaxis, problem.environment.yaxis
    x, y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    static_map = np.meshgrid(x, y)

    for _ in range(100):
        path =
