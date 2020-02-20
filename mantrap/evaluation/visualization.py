from typing import Any, Dict, Union

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import torch

from mantrap.constants import agent_speed_max
from mantrap.simulation.simulation import Simulation
from mantrap.utility.maths import Derivative2
from mantrap.utility.shaping import check_state
from mantrap.utility.utility import build_trajectory_from_path


def visualize_optimization(optimization_log: Dict[str, Any], env: Simulation, file_path: str):
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

    fig = plt.figure(figsize=(15, 15), constrained_layout=True)
    grid = plt.GridSpec(len(vis_keys) + 3, len(vis_keys), wspace=0.4, hspace=0.3, figure=fig)
    axs = list()
    axs.append(fig.add_subplot(grid[: len(vis_keys), :]))
    axs.append(fig.add_subplot(grid[-3, :]))
    axs.append(fig.add_subplot(grid[-2, :]))
    for j, _ in enumerate(vis_keys):
        axs.append(fig.add_subplot(grid[-1, j]))

    def update(k):
        time_axis = np.linspace(env.sim_time, env.sim_time + horizon * env.dt, num=horizon)

        x2_np = optimization_log["x"][k]
        x2 = torch.from_numpy(x2_np)
        ego_traj = build_trajectory_from_path(x2, dt=env.dt, t_start=env.sim_time)
        ado_traj = env.predict(graph_input=ego_traj)

        plt.axis("off")
        for i in range(len(axs)):
            axs[i].cla()

        # Plot current and base solution in the scene. This includes the determined ego trajectory (x) as well as
        # the resulting ado trajectories based on some simulation.
        axs[0].plot(x2_np[:, 0], x2_np[:, 1], "-", color=env.ego.color, label="ego_current")
        _add_agent_representation(env.ego.state, env.ego.color, "ego", ax=axs[0])
        # Plot current and base resulting simulated ado trajectories in the scene.
        for i in range(env.num_ado_ghosts):
            i_ado, i_mode = env.ghost_to_ado_index(i)
            ado_id, ado_color = env.ado_ghosts[i].id, env.ado_ghosts[i].agent.color
            ado_pos = ado_traj[i_ado, i_mode, :, 0:2].detach().numpy()
            axs[0].plot(ado_pos[:, 0], ado_pos[:, 1], "--", color=ado_color, label=f"{ado_id}_current")
            _add_agent_representation(env.ado_ghosts[i].agent.state, ado_color, ado_id, ax=axs[0])
        axs[0].set_title(f"iGrad optimization - IPOPT step {k} - Horizon {horizon}")
        axs[0].set_xlim(env.axes[0])
        axs[0].set_ylim(env.axes[1])
        axs[0].grid()
        axs[0].legend()

        # Plot agent velocities for resulting solution vs base-line ego trajectory for current optimization step.
        ado_velocity_norm = np.linalg.norm(ado_traj[:, :, :, 2:4].detach().numpy(), axis=3)
        ego_velocity_norm = np.linalg.norm(ego_traj[:, 2:4].detach().numpy(), axis=1)
        for i in range(env.num_ado_ghosts):
            i_ado, i_mode = env.ghost_to_ado_index(i)
            ado_id, ado_color = env.ado_ghosts[i].id, env.ado_ghosts[i].agent.color
            axs[1].plot(time_axis, ado_velocity_norm[i_ado, i_mode, :], color=ado_color, label=f"{ado_id}_current")
        axs[1].plot(time_axis, ego_velocity_norm, color=env.ego.color, label="ego_current")
        axs[1].set_title("velocities [m/s]")
        axs[1].set_ylim(0, agent_speed_max)
        axs[1].grid()
        axs[1].legend()

        # Plot agent accelerations for resulting solution vs base-line ego trajectory for current optimization step.
        dd = Derivative2(horizon=horizon, dt=env.dt)
        ado_acceleration_norm = np.linalg.norm(dd.compute(ado_traj[:, :, :, 0:2]).detach().numpy(), axis=3)
        for i in range(env.num_ado_ghosts):
            i_ado, i_mode = env.ghost_to_ado_index(i)
            ado_id, ado_color = env.ado_ghosts[i].id, env.ado_ghosts[i].agent.color
            axs[2].plot(time_axis, ado_acceleration_norm[i_ado, i_mode, :], color=ado_color, label=f"{ado_id}_current")
        axs[2].set_title("accelerations [m/s^2]")
        axs[2].set_ylim(0, agent_speed_max / 2.0)
        axs[2].grid()
        axs[2].legend()

        # Plot several parameter describing the optimization process, such as objective value, gradient and
        # the constraints (primal) infeasibility.
        for i, vis_key in enumerate(vis_keys):
            for name, data in optimization_log.items():
                if vis_key not in name:
                    continue
                axs[i + 3].plot(optimization_log["iter_count"][:k], np.log(np.asarray(data[:k]) + 1e-8), label=name)
            axs[i + 3].set_title(f"log_{vis_key}")
            axs[i + 3].legend()
            axs[i + 3].grid()

        return axs

    anim = FuncAnimation(fig, update, frames=optimization_log["iter_count"][-1], interval=300)
    anim.save(f"{file_path}.gif", dpi=60, writer='imagemagick')


def _add_agent_representation(state: torch.Tensor, color: np.ndarray, name: Union[str, None], ax: plt.Axes):
    assert check_state(state, enforce_temporal=False), "state vector is invalid"
    state = state.detach().numpy()

    # Add circle for agent itself.
    ado_circle = plt.Circle(state[0:2], 0.2, color=color, clip_on=True)
    ax.add_artist(ado_circle)
    arrow_length = np.linalg.norm(state[2:4]) / agent_speed_max * 0.5

    # Add agent id description.
    if id is not None:
        ax.text(state[0], state[1], name, fontsize=8)

    # Add arrow for orientation and speed.
    rot = np.array([[np.cos(state[2]), -np.sin(state[2])], [np.sin(state[2]), np.cos(state[2])]])
    darrow = rot.dot(np.array([1, 0])) * arrow_length
    head_width = max(0.02, arrow_length / 10)
    plt.arrow(state[0], state[1], darrow[0], darrow[1], head_width=head_width, head_length=0.1, fc="k", ec="k")
    return ax
