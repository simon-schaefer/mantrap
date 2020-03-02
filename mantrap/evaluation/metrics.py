import torch

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
    inter_steps = 100 if "num_inter_points" not in metric_kwargs.keys() else metric_kwargs["num_inter_points"]

    ego_trajectory = metric_kwargs["ego_trajectory"]
    assert check_ego_trajectory(ego_trajectory, pos_only=True)
    ado_trajectories = metric_kwargs["ado_trajectories"]
    horizon = ego_trajectory.shape[0]
    num_modes = ado_trajectories.shape[1]
    assert check_trajectories(ado_trajectories, t_horizon=horizon, pos_only=True)

    minimal_distance = float("Inf")
    for t in range(1, horizon):
        ego_dense = straight_line(ego_trajectory[t - 1, 0:2], ego_trajectory[t, 0:2], steps=inter_steps)
        for ado_trajectory in ado_trajectories:
            for m in range(num_modes):
                ado_dense = straight_line(ado_trajectory[m, t - 1, 0:2], ado_trajectory[m, t, 0:2], steps=inter_steps)
                min_distance_current = torch.min(torch.norm(ego_dense - ado_dense, dim=1)).item()
                if min_distance_current < minimal_distance:
                    minimal_distance = min_distance_current

    return float(minimal_distance)


def metric_ego_effort(**metric_kwargs) -> float:
    """Determine the ego's control effort (acceleration).
    For calculating the control effort of the ego agent approximate the acceleration by assuming the acceleration
    between two points in discrete time t0 and t1 as linear, i.e. a_t = (v_t - v_{t-1}) / dt. Then accumulate the
    acceleration for the final score.

    :param: metric_kwargs: dictionary of results of one (!) testing run.
    """
    assert "ego_trajectory" in metric_kwargs.keys()
    assert check_ego_trajectory(metric_kwargs["ego_trajectory"])

    ego_trajectory = metric_kwargs["ego_trajectory"]
    effort_score = 0.0
    for t in range(2, ego_trajectory.shape[0]):
        dt = ego_trajectory[t, -1] - ego_trajectory[t - 1, -1]
        effort_score += torch.norm(ego_trajectory[t, 2:4] - ego_trajectory[t-1, 2:4]).item() / dt

    return float(effort_score)
