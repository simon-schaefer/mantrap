from typing import Tuple

import numpy as np

from mantrap.constants import sim_dt_default, planning_horizon_default
from mantrap.simulation.simulation import Simulation


def straight_line(
    sim: Simulation, goal: np.ndarray, t_horizon: int = planning_horizon_default, dt: float = sim_dt_default
) -> Tuple[np.ndarray, np.ndarray]:
    """Straight line from initial to target position with constant velocity as ego trajectory.
    :param sim: simulation environment.
    :param goal: goal state of ego containing goal position, (>=2).
    :param t_horizon: planning horizon i.e. length of ego trajectory.
    :param dt: simulation time-step [s].
    :return: ego and ado trajectories for straight-line ego movement.
    """
    ego_trajectory = np.vstack(
        (
            np.linspace(sim.ego.position[0], goal[0], t_horizon),
            np.linspace(sim.ego.position[1], goal[1], t_horizon),
            np.zeros(t_horizon),
            np.ones(t_horizon) * (goal[0] - sim.ego.position[0]) / (t_horizon * dt),
            np.ones(t_horizon) * (goal[1] - sim.ego.position[1]) / (t_horizon * dt),
            np.linspace(0, t_horizon, t_horizon),
        )
    ).T
    ado_trajectories = sim.predict(t_horizon=t_horizon, ego_trajectory=ego_trajectory)
    return ego_trajectory, ado_trajectories
