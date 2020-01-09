from typing import Dict, Tuple, Union

import numpy as np

from mantrap.utility.io import path_from_home_directory
from mantrap.utility.shaping import check_ado_trajectories


def load_eth(return_id_dict: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict[int, int]]]:
    trajectory_data = np.loadtxt(path_from_home_directory("datasets/eth/train/students001_train.txt"))
    ado_ids = np.unique(trajectory_data[:, 1]).astype(int)
    ado_id_dict = {ado_id: i for i, ado_id in enumerate(ado_ids)}
    time_steps = np.sort(np.unique(trajectory_data[:, 0]))
    ado_trajectories = np.zeros((ado_ids.size, 1, time_steps.size, 6))

    # Iterating through the dataset by going from one time-step to another, while assuming that the trajectory data
    # array is sorted (by increasing time). However, it saves computation time for looking for the array index of the
    # current time.
    time_index = 0
    for data in trajectory_data:
        t, ado_id, x, y = data
        if not np.isclose(t, time_steps[time_index]):
            time_index = time_index + 1
        ado_id_index = ado_id_dict[int(ado_id)]
        ado_trajectories[ado_id_index, 0, time_index, 0] = x
        ado_trajectories[ado_id_index, 0, time_index, 1] = y
        ado_trajectories[ado_id_index, 0, time_index, 5] = time_steps[time_index] * 0.001  # ms -> seconds

    assert check_ado_trajectories(ado_trajectories, num_ados=ado_ids.size, num_modes=1)

    if return_id_dict:
        return ado_trajectories, ado_id_dict
    else:
        return ado_trajectories
