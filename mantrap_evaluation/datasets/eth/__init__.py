import os
from typing import Dict, Tuple, Union

import mantrap
import numpy as np
import torch

from mantrap_evaluation.datasets.api import _create_environment


def scenario_eth(
    env_type: mantrap.environment.GraphBasedEnvironment.__class__,
    ego_type: mantrap.agents.Agent.__class__ = mantrap.agents.DoubleIntegratorDTAgent,
    t_dataset: float = 0.0,
    num_modes: int = 1
) -> Tuple[mantrap.environment.GraphBasedEnvironment, torch.Tensor, Union[Dict[str, torch.Tensor], None]]:
    """ETH - Computer Vision Lab - Pedestrian movement dataset at ETH.

    The argument `t_dataset` determines the time of the dateset (video) in which the pedestrian has to be present,
    in order to be taken into account for the created scenario.

    :param env_type: type of created environment.
    :param ego_type: type of created ego agent (robot).
    :param t_dataset: dataset starting time around which pedestrian should be used [s].
    :param num_modes: number of output modes.
    """
    # Dataset parameters.
    eth_dt = 0.4  # [s] time-step.

    # Adapt dataset time to be sure that it is evenly dividable by the dataset time-step. Since `eth_dt` is
    # small the scene is not fundamentally changed from the scene that the user originally asked for.
    t_dataset = np.divide(t_dataset, eth_dt) * eth_dt

    # Read trajectories information file from dataset into numpy array using `loadtxt` function.
    dataset_file = os.path.join("mantrap_evaluation", "datasets", "eth", "ewap_dataset", "seq_eth", "obsmat.txt")
    dataset_file = mantrap.utility.io.build_os_path(dataset_file)
    data = np.loadtxt(dataset_file)

    # Normalize file i.e. shift positions to mean over all position and reset time-step by subtracting the first
    # time-step. Also convert the frame number column to actual time-steps using the time-step given in the
    # dataset's documentation. For more information on how to read the dataset file read its README file.
    data[:, 0] = (data[:, 0] - data[0, 0]) * eth_dt / 6  # 6 frames between annotations
    mean_pos_x = np.mean(data[:, 2])
    data[:, 2] -= mean_pos_x
    mean_pos_y = np.mean(data[:, 4])
    data[:, 4] -= mean_pos_y
    assert 0 <= t_dataset <= data[-1, 0]  # assert that `t_dataset` is in range at all

    # Determine environment axes from min and max values.
    x_axis = (min(data[:, 2]), max(data[:, 2]))
    y_axis = (min(data[:, 4]), max(data[:, 4]))

    # Split the data by index by first finding all unique ids and then their data indexes.
    pedestrian_ids = np.unique(data[:, 1])
    ado_histories = []
    ado_ids = []
    ado_goals = []
    ado_ground_truths = {}
    column_order = [2, 4, 5, 7, 0]  # (px, py, vx, vy, time)
    for ado_id in pedestrian_ids:
        data_indexes = np.argwhere(np.isclose(data[:, 1], ado_id))  # is_close since index is numeric
        trajectory = data[data_indexes, column_order]

        # If pedestrian is not present at the demanded time, continue without adding it to output lists.
        t_start = float(trajectory[0, -1])
        t_end = float(trajectory[-1, -1])
        if t_start <= t_dataset <= t_end:
            # Perform checks on data per pedestrian (id) such as whether time axis is continuously.
            num = round((t_end - t_start) / eth_dt)
            assert np.all(np.isclose(trajectory[:, -1], np.linspace(t_start, t_end, num=num + 1)))

            # Sync trajectories by resetting all to t0 = t_dataset = 0.
            trajectory[:, -1] -= t_dataset
            history = trajectory[trajectory[:, -1] <= 0.0, :]
            future = trajectory[trajectory[:, -1] >= 0.0, :]

            # When everything is fine include pedestrian in result lists.
            ado_id = str(int(ado_id))
            ado_ids.append(ado_id)
            ado_histories.append(torch.from_numpy(history))
            ado_ground_truths[ado_id] = torch.from_numpy(future)
            ado_goals.append(torch.from_numpy(trajectory[-1, 0:2]))  # goal = last trajectory point

    # Create ego information, dependent on the input ego information.
    ego_state = torch.zeros(4) if ego_type is not None else None
    ego_goal = torch.zeros(2)

    # Create environment using api.
    env = _create_environment(
        env_type=env_type,
        config_name="eth",
        ado_histories=ado_histories,
        ado_ids=ado_ids,
        ego_type=ego_type,
        ego_state=ego_state,
        ado_goals=ado_goals,
        num_modes=num_modes,
        dt=eth_dt,  # environment time-delta
        x_axis=x_axis,
        y_axis=y_axis
    )

    return env, ego_goal, ado_ground_truths
