import torch

from mantrap.constants import agent_acc_max
from mantrap.utility.io import dict_value_or_default
from mantrap.utility.maths import Derivative2
from mantrap.utility.primitives import straight_line
from mantrap.utility.shaping import check_ego_trajectory, check_trajectories


def metric_minimal_distance(**metric_kwargs) -> float:
    """Determine the minimal distance between the robot and any agent.
    Therefore the function expects to get a robot trajectory and positions for every ado (ghost) at every point of time,
    to determine the minimal distance in the continuous time. In order to transform the discrete to continuous time
    trajectories it is assumed that the robot as well as the other agents move linearly, as a single integrator, i.e.
    neglecting accelerations, from one discrete time-step to another, so that it's positions can be interpolated
    in between using a first order interpolation method.

    :param: metric_kwargs: dictionary of results of one (!) testing run.
    """
    assert all([x in metric_kwargs.keys() for x in ["ego_trajectory", "ado_trajectories"]])
    n_inter = dict_value_or_default(metric_kwargs, key="num_inter_points", default=100)

    ego_trajectory = metric_kwargs["ego_trajectory"].detach()
    assert check_ego_trajectory(ego_trajectory, pos_only=True)
    ado_trajectories = metric_kwargs["ado_trajectories"].detach()
    horizon = ego_trajectory.shape[0]
    num_ados = ado_trajectories.shape[0]
    assert check_trajectories(ado_trajectories, t_horizon=horizon, pos_only=True, modes=1)

    minimal_distance = float("Inf")
    for t in range(1, horizon):
        ego_dense = straight_line(ego_trajectory[t - 1, 0:2], ego_trajectory[t, 0:2], steps=n_inter)
        for m in range(num_ados):
            ado_dense = straight_line(ado_trajectories[m, 0, t-1, 0:2], ado_trajectories[m, 0, t, 0:2], steps=n_inter)
            min_distance_current = torch.min(torch.norm(ego_dense - ado_dense, dim=1)).item()
            if min_distance_current < minimal_distance:
                minimal_distance = min_distance_current

    return float(minimal_distance)


def metric_ego_effort(**metric_kwargs) -> float:
    """Determine the ego's control effort (acceleration).

    For calculating the control effort of the ego agent approximate the acceleration by assuming the acceleration
    between two points in discrete time t0 and t1 as linear, i.e. a_t = (v_t - v_{t-1}) / dt. For normalization
    then compare the determined acceleration with the maximal acceleration the agent maximally would be capable of.
    The ego_effort score then is the ratio between the actual requested and maximally possible control effort.

    :param: metric_kwargs: dictionary of results of one (!) testing run.
    """
    assert all([x in metric_kwargs.keys() for x in ["ego_trajectory"]])
    max_acceleration = dict_value_or_default(metric_kwargs, key="max_acceleration", default=agent_acc_max)
    ego_trajectory = metric_kwargs["ego_trajectory"].detach()
    assert check_ego_trajectory(ego_trajectory)

    # Determine integral over ego acceleration (= ego speed). Similarly for single integrator ego type.
    ego_effort = 0.0
    max_effort = 0.0
    for t in range(1, ego_trajectory.shape[0]):
        dt = ego_trajectory[t, -1] - ego_trajectory[t - 1, -1]
        dd = Derivative2(dt=dt, horizon=2, velocity=True)
        ego_effort += torch.norm(dd.compute(ego_trajectory[t-1:t+1, 2:4])).item()
        max_effort += max_acceleration

    return float(ego_effort / max_effort)


def metric_ado_effort(**metric_kwargs) -> float:
    """Determine the ado's additional control effort introduced by the ego.

    For calculating the additional control effort of the ado agents their acceleration is approximately determined
    numerically and compared to the acceleration of the according ado in a scene without ego robot. Then accumulate
    the acceleration differences for the final score.

    :param: metric_kwargs: dictionary of results of one (!) testing run.
    """
    assert all([x in metric_kwargs.keys() for x in ["ado_trajectories", "env"]])
    ado_traj = metric_kwargs["ado_trajectories"].detach()
    t_horizon = ado_traj.shape[2]
    num_ados = ado_traj.shape[0]
    assert check_trajectories(ado_traj, modes=1)  # deterministic (!)

    # Copy environment to not alter passed env object when resetting its state. Also check whether the initial
    # state in the environment and the ado trajectory tensor are equal.
    env = metric_kwargs["env"].copy()
    for j, ghost in enumerate(env.ghosts):
        i_ado, i_mode = env.index_ghost_id(ghost_id=ghost.id)
        assert torch.all(torch.isclose(ado_traj[i_ado, i_mode, 0, :], ghost.agent.state_with_time))

    effort_score = 0.0
    for m in range(num_ados):
        for t in range(1, t_horizon):
            # Reset environment to last ado states.
            env.step_reset(ego_state_next=None, ado_states_next=ado_traj[:, :, t - 1, :].unsqueeze(dim=2))

            # Predicting ado trajectories without interaction for current state.
            ado_traj_wo = env.predict_wo_ego(t_horizon=2).detach()
            assert check_trajectories(ado_traj_wo, ados=num_ados, t_horizon=2)

            # Determine acceleration difference between actual and without scene w.r.t. ados.
            dt = ado_traj[m, :, t, -1] - ado_traj[m, :, t - 1, -1]
            dd = Derivative2(horizon=2, dt=dt, velocity=True)
            ado_acc = torch.norm(dd.compute(ado_traj[m, :, t-1:t+1, 2:4]))
            ado_acc_wo = torch.norm(dd.compute(ado_traj_wo[m, :, 0:2, 2:4]))

            # Accumulate L2 norm of difference in metric score.
            effort_score += torch.norm(ado_acc - ado_acc_wo).detach()

    return float(effort_score) / num_ados


def metric_directness(**metric_kwargs) -> float:
    """Determine how direct the robot is going from start to goal state.

    Metrics should be fairly independent to be really meaningful, however measuring the efficiency of the ego trajectory
    only based on the travel time from start to goad position is highly connected to the ego's control effort.
    Therefore the ratio of every ego velocity vector going in the goal direction is determined, and normalized by the
    number of time-steps.

    .. math:: score = \dfrac{\sum_t \overrightarrow{s}_t * \overrightarrow{v}_t}{T}

    :param: metric_kwargs: dictionary of results of one (!) testing run.
    """
    assert all([x in metric_kwargs.keys() for x in ["ego_trajectory", "goal"]])
    ego_trajectory = metric_kwargs["ego_trajectory"].detach()
    goal = metric_kwargs["goal"].float()
    assert check_ego_trajectory(ego_trajectory)
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

    return float(score) / t_horizon_until_goal
