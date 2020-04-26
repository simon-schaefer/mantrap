from copy import deepcopy
import math

import torch

from mantrap.agents.agent import Agent
from mantrap.utility.shaping import check_ego_path


"""torch: 167 ms ± 16.7 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)"""

def pi_controller(
    agent: Agent,
    path: torch.Tensor,
    max_sim_time: float,
    dtc: float,
    speed_reference: float = None,
    breaking_acc: float = None,
):
    assert check_ego_path(path)
    num_path_points = path.shape[0]
    max_sim_steps = int(max_sim_time / dtc)

    with torch.no_grad():
        # Controller parameters.
        speed_reference = speed_reference if speed_reference is not None else agent.speed_max
        breaking_acc = breaking_acc if breaking_acc is not None else agent.acceleration_max

        # Controller initialization - Create a speed profile along given path based on reference speed
        # and the position on the trajectory, i.e. the closer to the end the agent gets, the smaller
        # the profiled velocity should be.
        min_breaking_time = math.ceil((speed_reference / breaking_acc) / dtc)
        min_breaking_time = max(min_breaking_time, 1)
        velocity_profile = torch.ones(num_path_points) * speed_reference
        velocity_profile[-min_breaking_time:] = torch.linspace(speed_reference, end=0.0, steps=min_breaking_time)

        # Control loop - Iteratively find the nearest index on the path, determine the target path point
        # and compute/execute the control inputs for the agent. Repeat until either the maximal number of
        # iterations (maximal simulation time) or the end of the path has been reached.
        sim_time, sim_step = 0.0, 0
        nearest_index, target_index = None, 0
        sim_agent = deepcopy(agent)
        controls = torch.zeros((max_sim_steps + 1, 2))

        while max_sim_time > sim_time and num_path_points > target_index + 1:
            # Determine closest path index and compute look ahead distance based on the reference velocity
            # at this point. Thereby the robot does not go sidewards, but still is able to adapt to the
            # reference velocity over time.
            nearest_index = _closest_path_index(path, position=sim_agent.position, index_start=nearest_index)
            look_ahead = 2.0 + 0.1 * sim_agent.speed

            # Determine target path point.
            target_index = _index_in_distance(path, sim_agent.position, index_start=nearest_index, distance=look_ahead)
            target_point = path[target_index]
            target_speed = velocity_profile[nearest_index]

            dxy = target_point - sim_agent.position
            yaw = torch.atan2(sim_agent.velocity[1], sim_agent.velocity[0])

            v, vx, vy = sim_agent.speed, sim_agent.velocity[0], sim_agent.velocity[1]
            alpha = torch.atan2(dxy[1], dxy[0]) - yaw
            d_vel = 1.0 * (target_speed - sim_agent.speed)
            delta = torch.atan2(2.0 * 0.05 * torch.sin(alpha) / look_ahead, torch.ones(1))
            d_yaw = v / 0.05 * math.tan(delta)

            u = torch.tensor([0.5 * d_vel * vx / v - vy * d_yaw,
                             0.5 * d_vel * vy / v + vx * d_yaw])

            # Do simulation step, i.e. update agent with computed control input.
            u_executed, _ = sim_agent.update(action=u, dt=dtc)
            controls[sim_step, :] = u_executed
            sim_time, sim_step = sim_time + dtc, sim_step + 1

    return controls[:sim_step, :]


def _closest_path_index(path: torch.Tensor, position: torch.Tensor, index_start: int = None) -> int:
    if index_start is None:
        distances = torch.norm(position.unsqueeze(dim=0) - path, dim=1)
        closest_index = torch.argmin(distances, dim=0).item()

    else:
        num_path_points = path.shape[0] - 1
        index = index_start
        distance = torch.norm(position - path[index, :])
        while True:
            distance_next = torch.norm(position - path[index + 1, :])
            if torch.le(distance, distance_next):
                break
            index = index + 1 if (index + 1) < num_path_points else index
            distance = distance_next
        closest_index = index

    return closest_index


def _index_in_distance(path: torch.Tensor, position: torch.Tensor, distance: float, index_start: int = 0) -> int:
    num_path_points = path.shape[0] - 1
    target_index = index_start
    if target_index >= num_path_points:
        return num_path_points

    while distance > torch.norm(position - path[target_index]):
        if target_index + 1 > num_path_points:
            break  # not exceed goal
        target_index += 1

    return target_index
