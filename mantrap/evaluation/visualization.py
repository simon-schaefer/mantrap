import os
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from mantrap.constants import agent_speed_max
from mantrap.simulation.simulation import Simulation
from mantrap.utility.maths import Derivative2
from mantrap.utility.utility import build_trajectory_from_positions


def visualize_optimization(optimization_log: Dict[str, Any], env: Simulation, dir_path: str):
    assert "iter_count" in optimization_log.keys(), "iteration count must be provided in optimization dict"
    assert "x" in optimization_log.keys(), "trajectory data x must be provided in optimization dict"

    vis_keys = ["obj", "inf", "grad"]
    horizon = optimization_log["x"][0].shape[0]

    # For comparison in the visualization predict the behaviour of every agent in the scene for the base
    # trajectory, i.e. x0 the initial value trajectory.
    # x2_base_np = optimization_log["x"][0]
    # x2_base = torch.from_numpy(x2_base_np)
    # ego_traj_base = build_trajectory_from_positions(x2_base, dt=env.dt, t_start=env.sim_time)
    # ego_traj_base_np = ego_traj_base.detach().numpy()
    # ado_traj_base_np = env.predict(horizon, ego_trajectory=torch.from_numpy(x2_base_np)).detach().numpy()

    for k in range(1, optimization_log["iter_count"][-1]):
        time_axis = np.linspace(env.sim_time, env.sim_time + horizon * env.dt, num=horizon)

        x2_np = optimization_log["x"][k]
        x2 = torch.from_numpy(x2_np)
        ego_traj = build_trajectory_from_positions(x2, dt=env.dt, t_start=env.sim_time)
        ado_traj_np = env.predict(horizon, ego_trajectory=ego_traj).detach().numpy()

        fig = plt.figure(figsize=(15, 15), constrained_layout=True)
        plt.title(f"iGrad optimization - IPOPT step {k} - Horizon {horizon}")
        plt.axis("off")
        grid = plt.GridSpec(len(vis_keys) + 3, len(vis_keys), wspace=0.4, hspace=0.3)

        # Plot current and base solution in the scene. This includes the determined ego trajectory (x) as well as
        # the resulting ado trajectories based on some simulation.
        ax = fig.add_subplot(grid[: len(vis_keys), :])
        ax.plot(x2_np[:, 0], x2_np[:, 1], "-", color=env.ego.color, label="ego_current")
        _add_agent_representation(env.ego.state.detach().numpy(), env.ego.color, "ego", ax=ax)
        # ax.plot(x2_base_np[:, 0], x2_base_np[:, 1], "o", color=env.ego.color, label="ego_base")
        # Plot current and base resulting simulated ado trajectories in the scene.
        for m in range(env.num_ados):
            ado_id, ado_color = env.ados[m].id, env.ados[m].color
            ado_pos = ado_traj_np[m, 0, :, 0:2]
            # ado_pos_base = ado_traj_base_np[m, 0, :, 0:2]
            ax.plot(ado_pos[:, 0], ado_pos[:, 1], "--", color=ado_color, label=f"{ado_id}_current")
            # ax.plot(ado_pos_base[:, 0], ado_pos_base[:, 1], "o", color=ado_color, label=f"{ado_id}_base")
            _add_agent_representation(env.ados[m].state.detach().numpy(), ado_color, ado_id, ax=ax)
        ax.set_xlim(env.axes[0])
        ax.set_ylim(env.axes[1])
        plt.grid()
        plt.legend()

        # Plot agent velocities for resulting solution vs base-line ego trajectory for current optimization step.
        ax = fig.add_subplot(grid[-3, :])
        ado_velocity_norm = np.linalg.norm(ado_traj_np[:, :, :, 3:5], axis=3)
        # ado_velocity_base_norm = np.linalg.norm(ado_traj_base_np[:, :, :, 3:5], axis=3)
        ego_velocity_norm = np.linalg.norm(ego_traj[:, 3:5], axis=1)
        # ego_velocity_base_norm = np.linalg.norm(ego_traj_base_np[:, 3:5], axis=1)
        for m in range(env.num_ados):
            ado_id, ado_color = env.ados[m].id, env.ados[m].color
            ax.plot(time_axis, ado_velocity_norm[m, 0, :], color=ado_color, label=f"{ado_id}_current")
            # ax.plot(time_axis, ado_velocity_base_norm[m, 0, :], "--", color=ado_color, label=f"{ado_id}_base")
        ax.plot(time_axis, ego_velocity_norm, color=env.ego.color, label="ego_current")
        # ax.plot(time_axis, ego_velocity_base_norm, "--", color=env.ego.color, label="ego_base")
        ax.set_title("velocities [m/s]")
        ax.set_ylim(0, agent_speed_max)
        plt.grid()
        plt.legend()

        # Plot agent accelerations for resulting solution vs base-line ego trajectory for current optimization step.
        ax = fig.add_subplot(grid[-2, :])
        dd = Derivative2(horizon=horizon, dt=env.dt)
        ado_acceleration_norm = np.linalg.norm(dd.compute(ado_traj_np[:, :, :, 0:2]), axis=3)
        # ado_base_acceleration_norm = np.linalg.norm(dd.compute(ado_traj_base_np[:, :, :, 0:2]), axis=3)
        for m in range(env.num_ados):
            ado_id, ado_color = env.ados[m].id, env.ados[m].color
            ax.plot(time_axis, ado_acceleration_norm[m, 0, :], color=ado_color, label=f"{ado_id}_current")
            # ax.plot(time_axis, ado_base_acceleration_norm[m, 0, :], "--", color=ado_color, label=f"{ado_id}_base")
        ax.set_title("accelerations [m/s^2]")
        ax.set_ylim(0, agent_speed_max / 2.0)
        plt.grid()
        plt.legend()

        # Plot several parameter describing the optimization process, such as objective value, gradient and
        # the constraints (primal) infeasibility.
        for i, vis_key in enumerate(vis_keys):
            ax = fig.add_subplot(grid[-1, i])
            for name, data in optimization_log.items():
                if vis_key not in name:
                    continue
                ax.plot(optimization_log["iter_count"][:k], np.log(np.asarray(data[:k]) + 1e-8), label=name)
            ax.set_title(f"log_{vis_key}")
            plt.legend()
            plt.grid()

        plt.savefig(os.path.join(dir_path, f"{k}.png"), dpi=60)
        plt.close()


def _add_agent_representation(state: np.ndarray, color: np.ndarray, name: Union[str, None], ax: plt.Axes):
    assert state.size == 5 or state.size == 6, "state must be of size 5 or 6 (x, y, theta, vx, vy, t)"
    arrow_length = np.linalg.norm(state[3:5]) / agent_speed_max * 0.5

    # Add circle for agent itself.
    ado_circle = plt.Circle(state[0:2], 0.2, color=color, clip_on=True)
    ax.add_artist(ado_circle)

    # Add agent id description.
    if id is not None:
        ax.text(state[0], state[1], name, fontsize=8)

    # Add arrow for orientation and speed.
    rot = np.array([[np.cos(state[2]), -np.sin(state[2])], [np.sin(state[2]), np.cos(state[2])]])
    darrow = rot.dot(np.array([1, 0])) * arrow_length
    head_width = max(0.02, arrow_length / 10)
    plt.arrow(state[0], state[1], darrow[0], darrow[1], head_width=head_width, head_length=0.1, fc="k", ec="k")
    return ax
