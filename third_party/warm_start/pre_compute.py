import os
import typing

import mantrap
import torch


def generate_random_scene(pos_bounds: typing.Tuple[float, float], vel_bounds: typing.Tuple[float, float]
                          ) -> typing.Tuple[mantrap.environment.PotentialFieldEnvironment, torch.Tensor]:
    """Generate random scenario with one pedestrian and the robot.

    :param pos_bounds: bounds of position sampling, both for robot position, goal and pedestrian position.
    :param vel_bounds: bounds of velocity sampling for robot.
    :returns random one-pedestrian environment (PotentialField), robot goal position
    """
    ego_position = pos_bounds[0] + torch.rand(2) * (pos_bounds[1] - pos_bounds[0])
    ego_velocity = vel_bounds[0] + torch.rand(2) * (vel_bounds[1] - vel_bounds[0])
    ado_position = pos_bounds[0] + torch.rand(2) * (pos_bounds[1] - pos_bounds[0])
    ego_goal = pos_bounds[0] + torch.rand(2) * (pos_bounds[1] - pos_bounds[0])

    env = mantrap.environment.PotentialFieldEnvironment(ego_position=ego_position, ego_velocity=ego_velocity,
                                                        dt=mantrap.constants.ENV_DT_DEFAULT)
    env.add_ado(position=ado_position, velocity=torch.rand(2))
    return env, ego_goal


def solve_random_scenario(num_scenarios: int = mantrap.constants.WARM_START_PRE_COMPUTATION_NUM,
                          solution_horizon: int = mantrap.constants.WARM_START_PRE_COMPUTATION_HORIZON
                          ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Randomly sample and solve scenarios, based on `IPOPTSolver` and `PotentialFieldEnvironment`.

    The simplified simulation environment `PotentialFieldEnvironment` is used in order to trade-off between
    the number of pre-computed scenarios (and therefore quality of match) and the meaningfulness of the
    pre-computed solution.
    """
    encodings = torch.zeros(num_scenarios, 4)  # 4 = encoding lengths
    solutions = torch.zeros(num_scenarios, solution_horizon, 2)  # 2 = robot control input size

    pos_bounds = mantrap.constants.ENV_X_AXIS_DEFAULT  # assuming isotropic bounds (x = y)
    vel_bounds = (0.0, mantrap.constants.ROBOT_SPEED_MAX)  # assuming isotropic bounds (vx = vy)
    for n in range(num_scenarios):
        print(f"Pre-Computation step = {n} / {num_scenarios}")
        env, ego_goal = generate_random_scene(pos_bounds=pos_bounds, vel_bounds=vel_bounds)
        solver = mantrap.solver.IPOPTSolver(env=env, goal=ego_goal, is_logging=False, is_debug=False)
        ego_trajectory, _ = solver.solve(time_steps=solution_horizon)

        # Store scenario encoding in database.
        encodings[n, :] = solver.encode()

        # Store solution in database. In case the robot arrived earlier than expected, stack
        # zeros at the remaining entries of the database.
        ego_controls = env.ego.roll_trajectory(ego_trajectory, dt=env.dt)
        diff_length = solution_horizon - ego_controls.shape[0]
        if diff_length > 0:
            ego_controls = torch.cat((ego_controls, torch.zeros((diff_length, 2))))
        solutions[n, :, :] = ego_controls

    return encodings, solutions


if __name__ == '__main__':
    encodings_db, solutions_db = solve_random_scenario()

    enc_file, solution_file = mantrap.constants.WARM_START_PRE_COMPUTATION_FILE
    output_directory = mantrap.utility.io.build_os_path(os.path.join("third_party", "warm_start"))
    torch.save(encodings_db, os.path.join(output_directory, enc_file))
    torch.save(solutions_db, os.path.join(output_directory, solution_file))
