import inspect
import logging
import sys
import time
import typing

import mantrap
import pandas
import torch


def evaluate(solver: mantrap.solver.base.TrajOptSolver, time_steps: int = 10, label: str = None,
             num_tests: int = 40, mean_df: bool = True, **solve_kwargs
             ) -> typing.Tuple[pandas.DataFrame, torch.Tensor, torch.Tensor]:
    """Evaluate the solver performance in current set configuration using MonteCarlo testing.

    For evaluation solve an N-step trajectory optimization problem, given additional solver-kwargs, which
    can be additionally passed. Then compute each metric value and return the results as `pd.DataFrame`.

    :param solver: solver to evaluation (has to be of `TrajOptSolver` class).
    :param time_steps: number of time-steps to solve for evaluation.
    :param label: label of evaluation in resulting data-frame, by default the log-name of the solver.
    :param num_tests: number of monte-carlo tests.
    :param mean_df: return the mean data-frame over all experiments.
    :param solve_kwargs: additional kwargs for `solve()` method.s
    """
    label = solver.log_name if label is None else label
    logging.info(f"Start evaluation {label} ...")

    # Solve internal optimization problem (measure average run-time).
    eval_df = pandas.DataFrame()
    ego_trajectories, ado_trajectories = [], []
    for k in range(num_tests):
        logging.info(f"Evaluation {label}: [{k}/{num_tests}] ...")

        start_time = time.time()
        ego_trajectory_k, ado_trajectories_k = solver.solve(time_steps=time_steps, **solve_kwargs)
        solve_time = time.time() - start_time
        ego_trajectories.append(ego_trajectory_k)
        ado_trajectories.append(ado_trajectories_k)

        # Build a dictionary of all metric functions listed in current file.
        metrics = {name.replace("metric_", ""): obj for name, obj in inspect.getmembers(sys.modules[__name__])
                   if (inspect.isfunction(obj) and name.startswith("metric"))}

        # Evaluate these metrics and build output data-frame.
        eval_dict = {}
        for name, metric_function in metrics.items():
            eval_dict[name] = metric_function(ego_trajectory=ego_trajectory_k,
                                              ado_trajectories=ado_trajectories_k,
                                              env=solver.env, goal=solver.goal)
        eval_dict["runtime[s]"] = solve_time / time_steps
        eval_df = eval_df.append(pandas.DataFrame(eval_dict, index=[k]))

    # ego_trajectories = torch.stack(ego_trajectories)
    # ado_trajectories = torch.stack(ado_trajectories).transpose(0, 1)
    # return eval_df_mean, ego_trajectories, ado_trajectories

    if mean_df:
        return eval_df.mean().to_frame(name=label).transpose(), None, None
    else:
        return eval_df, None, None


#######################################################################################################################
# Metric definitions ##################################################################################################
#######################################################################################################################
def metric_minimal_distance(
    ego_trajectory: torch.Tensor, ado_trajectories: torch.Tensor, num_inter_points: int = 100, **unused
) -> float:
    """Determine the minimal distance between the robot and any agent (minimal separation distance).

    Therefore the function expects to get a robot trajectory and positions for every ado at every point of time,
    to determine the minimal distance in the continuous time. In order to transform the discrete to continuous time
    trajectories it is assumed that the robot as well as the other agents move linearly, as a single integrator, i.e.
    neglecting accelerations, from one discrete time-step to another, so that it's positions can be interpolated
    in between using a first order interpolation method.

    :param ego_trajectory: trajectory of ego (t_horizon, 5).
    :param ado_trajectories: trajectories of ados (num_ados, num_modes, t_horizon, 5).
    :param num_inter_points: number of interpolation points between each time-step.
    """
    ego_trajectory = ego_trajectory.detach()
    assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, pos_only=True)
    ado_trajectories = ado_trajectories.detach()
    t_horizon = ego_trajectory.shape[0]
    num_ados = ado_trajectories.shape[0]
    assert mantrap.utility.shaping.check_ado_trajectories(ado_trajectories, t_horizon=t_horizon)

    minimal_distance = float("Inf")
    for t in range(1, t_horizon):
        ego_position_t0, ego_position_t1 = ego_trajectory[t - 1, 0:2], ego_trajectory[t, 0:2]
        ego_dense = mantrap.utility.maths.straight_line(ego_position_t0, ego_position_t1, steps=num_inter_points)

        for m in range(num_ados):
            ado_position_t0 = ado_trajectories[m, t-1, 0, 0:2]
            ado_position_t1 = ado_trajectories[m, t, 0, 0:2]
            ado_dense = mantrap.utility.maths.straight_line(ado_position_t0, ado_position_t1, steps=num_inter_points)

            min_distance_current = torch.min(torch.norm(ego_dense - ado_dense, dim=1)).item()
            if min_distance_current < minimal_distance:
                minimal_distance = min_distance_current

    return float(minimal_distance)


def metric_ego_effort(ego_trajectory: torch.Tensor, max_acceleration: float = mantrap.constants.ROBOT_ACC_MAX, **unused
                      ) -> float:
    """Determine the ego's control effort (acceleration).

    For calculating the control effort of the ego agent approximate the acceleration by assuming the acceleration
    between two points in discrete time t0 and t1 as linear, i.e. a_t = (v_t - v_{t-1}) / dt. For normalize
    then compare the determined acceleration with the maximal possible control effort.

    .. math:: score = \\frac{\\sum at}{\\sum a_{max}}

    :param ego_trajectory: trajectory of ego (t_horizon, 5).
    :param max_acceleration: maximal (possible) acceleration of ego robot.
    """
    ego_trajectory = ego_trajectory.detach()
    assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory)

    # Determine integral over ego acceleration (= ego speed). Similarly for single integrator ego type.
    ego_effort = 0.0
    max_effort = 0.0
    for t in range(1, ego_trajectory.shape[0]):
        dt = float(ego_trajectory[t, -1] - ego_trajectory[t - 1, -1])
        acc = mantrap.utility.maths.derivative_numerical(ego_trajectory[t-1:t+1, 2:4], dt=dt)
        ego_effort += torch.norm(acc).item()
        max_effort += max_acceleration

    return float(ego_effort / max_effort)


def metric_ado_effort(env: mantrap.environment.base.GraphBasedEnvironment, ado_trajectories: torch.Tensor, **unused
                      ) -> float:
    """Determine the ado's additional control effort introduced by the ego.

    For calculating the additional control effort of the ado agents their acceleration is approximately determined
    numerically and compared to the acceleration of the according ado in a scene without ego robot. Then accumulate
    the acceleration differences for the final score.

    :param ado_trajectories: trajectories of ados (num_ados, num_modes, t_horizon, 5).
    :param env: simulation environment (is copied within function, so not altered).
    """
    ado_trajectories = ado_trajectories.detach()
    t_horizon = ado_trajectories.shape[1]
    num_ados = ado_trajectories.shape[0]
    num_modes = ado_trajectories.shape[2]
    assert mantrap.utility.shaping.check_ado_trajectories(ado_trajectories)  # deterministic (!)

    # Copy environment to not alter passed env object when resetting its state. Also check whether the initial
    # state in the environment and the ado trajectory tensor are equal.
    env_metric = env.copy()
    assert env_metric.same_initial_conditions(other=env)

    effort_score = 0.0
    for m in range(num_ados):
        for t in range(1, t_horizon):
            # Reset environment to last ado states.
            env_metric.step_reset(ego_next=None, ado_next=ado_trajectories[:, t - 1, 0, :])

            # Predicting ado trajectories without interaction for current state.
            ado_trajectory_wo = env_metric.predict_wo_ego(t_horizon=1).detach()
            assert mantrap.utility.shaping.check_ado_trajectories(ado_trajectory_wo, ados=num_ados, t_horizon=2)

            # Determine acceleration difference between actual and without scene w.r.t. ados.
            dt = env_metric.dt
            for m_mode in range(num_modes):
                ado_acc = mantrap.utility.maths.derivative_numerical(ado_trajectories[m, t-1:t+1, m_mode, 2:4], dt=dt)
                ado_acc = torch.norm(ado_acc)
                ado_acc_wo = mantrap.utility.maths.derivative_numerical(ado_trajectory_wo[m, 0:2, m_mode, 2:4], dt=dt)
                ado_acc_wo = torch.norm(ado_acc_wo)

                # Accumulate L2 norm of difference in metric score.
                effort_score += (torch.norm(ado_acc - ado_acc_wo)).detach()

    return float(effort_score) / num_ados / num_modes


def metric_directness(ego_trajectory: torch.Tensor, goal: torch.Tensor, **unused) -> float:
    """Determine how direct the robot is going from start to goal state.

    Metrics should be fairly independent to be really meaningful, however measuring the efficiency of the ego trajectory
    only based on the travel time from start to goad position is highly connected to the ego's control effort.
    Therefore the ratio of every ego velocity vector going in the goal direction is determined, and normalized by the
    number of time-steps.

    .. math:: score = \\frac{\\sum_t \\overrightarrow{s}_t * \\overrightarrow{v}_t}{T}

    :param ego_trajectory: trajectory of ego (t_horizon, 5).
    :param goal: optimization goal state (may vary in size, but usually 2D position).
    """
    ego_trajectory = ego_trajectory.detach()
    assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory)
    goal = goal.float()
    t_horizon = ego_trajectory.shape[0]

    score = 0.0
    t_horizon_until_goal = 0
    for t in range(t_horizon):
        vt = ego_trajectory[t, 2:4]
        st = (goal - ego_trajectory[t, 0:2])
        if torch.norm(vt) < 1e-6 or torch.norm(st) < 1e-6:
            continue
        score += (vt / torch.norm(vt)).matmul((st / torch.norm(st)))
        t_horizon_until_goal += 1

    return float(score) / t_horizon_until_goal if abs(score) > 1e-3 else 0.0


def metric_final_distance(ego_trajectory: torch.Tensor, goal: torch.Tensor, **unused) -> float:
    """Determine the final distance between ego and its goal position.

    For normalize divide the final distance by the initial distance. Scores larger than 1 mean, that
    the robot is more distant from the goal than at the beginning.

    .. math:: score = ||x_T - g||_2 / ||x_0 - g||_2

    :param ego_trajectory: trajectory of ego (t_horizon, 5).
    :param goal: optimization goal state (may vary in size, but usually 2D position).
    """
    ego_trajectory = ego_trajectory.detach()
    assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory)
    goal = goal.float()
    distance_init = torch.norm(ego_trajectory[0, 0:2] - goal).item()
    distance_init = max(distance_init, 1e-6)  # avoid 0 division error
    distance_final = torch.norm(ego_trajectory[-1, 0:2] - goal).item()
    return distance_final / distance_init


def metric_extra_time(ego_trajectory: torch.Tensor, env: mantrap.environment.base.GraphBasedEnvironment, **unused
                      ) -> float:
    """Determine extra time to reach goal destination compared to direct path.

    Compare the derived ego trajectory travel time with the time it would need to get to the goal, which is
    defined as the last point of the ego trajectory, directly. To compute the direct path, solve the simplified
    optimization formulation consisting of goal objective and dynamics constraints only. The parameters
    other than the goal position do not matter for solving this simplified formulation.

    :param ego_trajectory: trajectory of ego (t_horizon, 5).
    """
    ego_trajectory = ego_trajectory.detach()
    assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory)

    goal = ego_trajectory[-1, 0:2]
    max_time_steps = ego_trajectory.shape[0]
    modules_hard = [mantrap.modules.GoalNormModule, mantrap.modules.SpeedLimitModule]
    solver_hard = mantrap.solver.IPOPTSolver(env=env, goal=goal, modules=modules_hard)

    ego_trajectory_straight, _ = solver_hard.solve(time_steps=max_time_steps)
    straight_time_steps = ego_trajectory_straight.shape[0]
    return (max_time_steps - straight_time_steps) * env.dt
