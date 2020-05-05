import math

import numpy as np
import torch

import mantrap.agents
import mantrap.utility.shaping


def p_ahead_controller(
    agent: mantrap.agents.base.DTAgent,
    path: torch.Tensor,
    max_sim_time: float,
    dtc: float,
    speed_reference: float = None,
    breaking_acc: float = None,
):
    """Look-ahead based P-Controller inspired by pure pursuit controller implemented in PythonRobotics library.

    This controller separates the optimization of "steering" and acceleration input and iteratively determines the
    control input to stay on the given path by alternating between calculating u (control) and simulating the
    agent's behaviour given that input. Therefore in every step first the nearest point, which is the point on
    the given path which is closest to the state of the agent, and the target point, the point it should steer
    to are computed. Given these points, the current agent's state and a reference speed profile along the path
    (constant speed, linearly breaking at the end) some control input is determined which maneuvers the agent to
    the target point.

    The algorithm is similar to the pure pursuit algorithm but using generalized dynamics. Due to high amount of
    repetitions, the small degree of dimensionality (assuming 2D) and since no gradient is required, the scalar
    implementation using native python types only is preferred. Compared to a PyTorch based implementation
    (167 ms ± 16.7 ms) it is way more efficient (5.17 ms ± 65.6 µs).

    :param agent: agent to control (including its state and dynamics).
    :param path: path positions to track (N, 2).
    :param max_sim_time: maximal time to forward simulate with dtc simulation time-steps [s].
    :param dtc: control simulation time-steps (output controls will be discretized using dtc).
    :param speed_reference: reference speed (constant phase of speed profile), max agent's speed per default [m/s].
    :param breaking_acc: maximal acceleration during breaking, max agent's acceleration per default [m/s^2].
    """
    assert mantrap.utility.shaping.check_ego_path(path)
    num_path_points = path.shape[0]

    # If the path length is one (aka greedy look-ahead) we steer to the next point directly.
    if num_path_points == 1:
        px, py, vx, vy = agent.state.detach().numpy()
        target_point = path.flatten().detach().numpy().tolist()

        _, (ux, uy) = agent.go_to_point(
            (px, py, vx, vy), target_point=target_point, speed=speed_reference, dt=dtc
        )

        controls = torch.tensor([ux, uy]).view(1, 2)
        assert mantrap.utility.shaping.check_ego_controls(controls, t_horizon=1)
        return controls.float()

    # Controller parameters.
    speed_reference = speed_reference if speed_reference is not None else agent.speed_max
    breaking_acc = breaking_acc if breaking_acc is not None else agent.acceleration_max
    assert speed_reference > 0 and breaking_acc > 0

    # Controller initialization - Create a speed profile along given path based on reference speed
    # and the position on the trajectory, i.e. the closer to the end the agent gets, the smaller
    # the profiled velocity should be.
    min_breaking_time = math.ceil((speed_reference / breaking_acc) / dtc)
    min_breaking_time = max(min_breaking_time, 1)
    velocity_profile = np.ones(num_path_points) * speed_reference
    velocity_profile[-min_breaking_time:] = np.linspace(speed_reference, 0.0, num=min_breaking_time)

    # Control loop - Iteratively find the nearest index on the path, determine the target path point
    # and compute/execute the control inputs for the agent. Repeat until either the maximal number of
    # iterations (maximal simulation time) or the end of the path has been reached.
    sim_time = 0.0
    nearest_index, target_index = None, 0

    cx = path[:, 0].detach().tolist()
    cy = path[:, 1].detach().tolist()

    px = float(agent.position[0].item())
    py = float(agent.position[1].item())
    vx = float(agent.velocity[0].item())
    vy = float(agent.velocity[1].item())
    v = math.hypot(vx, vy)

    controls = []

    while max_sim_time > sim_time and num_path_points > target_index + 1:
        # Determine closest path index and compute look ahead distance based on the reference velocity
        # at this point. Thereby the robot does not go sidewards, but still is able to adapt to the
        # reference velocity over time.
        nearest_index = _closest_path_index(cx, cy, px, py, index_start=nearest_index)
        look_ahead = 2.0 + 0.1 * v

        # Determine target path point.
        target_index = _index_in_distance(cx, cy, px, py, index=nearest_index, distance=look_ahead)
        target_speed = velocity_profile[nearest_index]

        # Compute acceleration and steering using pure pursuit formulas.
        (px, py, vx, vy), (ux, uy) = agent.go_to_point(
            (px, py, vx, vy), target_point=(cx[target_index], cy[target_index]), speed=target_speed, dt=dtc
        )
        controls.append((ux, uy))
        sim_time = sim_time + dtc

    # Convert controls to torch tensor, validate and return them.
    controls = torch.from_numpy(np.array(controls))
    assert mantrap.utility.shaping.check_ego_controls(controls, t_horizon=int(sim_time / dtc))
    return controls.float()


def _closest_path_index(cx, cy, px, py, index_start: int = None) -> int:
    if index_start is None:
        dx = [px - icx for icx in cx]
        dy = [py - icy for icy in cy]
        closest_index = np.argmin(np.hypot(dx, dy))

    else:
        index = index_start
        num_path_points = len(cx)
        distance = math.hypot(px - cx[index], py - cy[index])
        while index + 1 < num_path_points:
            distance_next = math.hypot(px - cx[index + 1], py - cy[index + 1])
            if distance <= distance_next:
                break
            index = index + 1 if (index + 1) < num_path_points else index
            distance = distance_next
        closest_index = index

    return closest_index


def _index_in_distance(cx, cy, px, py, distance: float, index: int = 0) -> int:
    target_index = index
    num_path_points = len(cx)
    while distance > math.hypot(px - cx[target_index], py - cy[target_index]):
        if (target_index + 1) >= num_path_points:
            break  # not exceed goal
        target_index += 1
    return target_index
