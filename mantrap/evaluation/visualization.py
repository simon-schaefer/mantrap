import logging
from typing import Any, Dict, List, Union

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import torch

from mantrap.simulation.simulation import GraphBasedSimulation
from mantrap.utility.maths import Derivative2
from mantrap.utility.shaping import check_state


def visualize_scenes(ego_opt_planned: torch.Tensor, ado_traj: torch.Tensor, env: GraphBasedSimulation, file_path: str):
    fig, ax = plt.subplots(figsize=(15, 15), constrained_layout=True)
    ado_traj_wo = env.predict_wo_ego(t_horizon=ado_traj.shape[0])

    def update(k):
        ego_traj = ego_opt_planned[k]

        ax.cla()

        # Plot current and base solution in the scene. This includes the determined ego trajectory (x) as well as
        # the resulting ado trajectories based on some simulation.
        ax.plot(ego_traj[:, 0].detach().numpy(), ego_traj[:, 1].detach().numpy(), "-", color=env.ego.color, label="ego")
        _add_agent_representation(ego_traj[0, :], env.ego.color, "ego", ax=ax)

        # Plot current and base resulting simulated ado trajectories in the scene.
        for i in range(env.num_ado_ghosts):
            ia, im = env.ghost_to_ado_index(i)
            ado_id, ado_color = env.ado_ghosts[i].id, env.ado_ghosts[i].agent.color
            _add_agent_representation(ado_traj[k, ia, im, 0, :], ado_color, ado_id, ax=ax)
            plt.plot(ado_traj[k, ia, im, :, 0], ado_traj[k, ia, im, :, 1], "*-", color=ado_color, label=ado_id)
            plt.plot(ado_traj_wo[ia, 0, k, 0], ado_traj_wo[ia, 0, k, 1], "--", color=ado_color, label=f"{ado_id}_wo")

        ax.set_xlim(env.axes[0])
        ax.set_ylim(env.axes[1])
        ax.grid()
        ax.legend()
        ax.set_title(f"Solving - Step {k}")

        return ax

    anim = FuncAnimation(fig, update, frames=ego_opt_planned.shape[0], interval=300)
    anim.save(f"{file_path}.gif", dpi=60, writer='imagemagick')


def visualize_optimization(
    log: Dict[str, Any],
    env: GraphBasedSimulation,
    file_path: str,
    last_only: bool = False,
    vis_keys: List[str] = None
):
    assert all([key in log.keys() for key in ["iter_count", "x4", "x4_trials"]])

    vis_keys_default = [x for x in ["obj", "inf", "grad"] if any([x in key for key in log.keys()])]
    vis_keys = vis_keys_default if vis_keys is None else vis_keys
    horizon = log["x4"][0].shape[0]

    fig = plt.figure(figsize=(15, 15), constrained_layout=True)
    grid = plt.GridSpec(len(vis_keys) + 4, max(len(vis_keys), 1), wspace=0.4, hspace=0.3, figure=fig)
    axs = list()
    axs.append(fig.add_subplot(grid[: max(len(vis_keys), 1), :]))
    axs.append(fig.add_subplot(grid[max(len(vis_keys), 1), :]))
    axs.append(fig.add_subplot(grid[max(len(vis_keys), 1) + 1, :]))
    axs.append(fig.add_subplot(grid[max(len(vis_keys), 1) + 2, :]))
    for j, _ in enumerate(vis_keys):
        axs.append(fig.add_subplot(grid[max(len(vis_keys), 1) + 3, j]))

    def update(k):
        time_axis = np.linspace(env.sim_time, env.sim_time + horizon * env.dt, num=horizon)

        x4 = log["x4"][k]
        ado_traj = env.predict_w_trajectory(trajectory=x4)
        ado_traj_wo = env.predict_wo_ego(t_horizon=x4.shape[0])

        plt.axis("off")
        for i in range(len(axs)):
            axs[i].cla()

        # Plot current and base solution in the scene. This includes the determined ego trajectory (x) as well as
        # the resulting ado trajectories based on some simulation.
        ego_trajectory_np = x4.detach().numpy()
        axs[0].plot(ego_trajectory_np[:, 0], ego_trajectory_np[:, 1], "-", color=env.ego.color, label="ego")
        _add_agent_representation(env.ego.state, env.ego.color, "ego", ax=axs[0])

        ego_traj_trials = log["x4_trials"][k]
        for x4_trial in ego_traj_trials:
            x4_trial_np = x4_trial.detach().numpy()
            axs[0].plot(x4_trial_np[:, 0], x4_trial_np[:, 1], "--", color=env.ego.color, alpha=0.02)

        # Plot current and base resulting simulated ado trajectories in the scene.
        for i in range(env.num_ado_ghosts):
            i_ado, i_mode = env.ghost_to_ado_index(i)
            ado_id, ado_color = env.ado_ghosts[i].id, env.ado_ghosts[i].agent.color
            ado_pos = ado_traj[i_ado, i_mode, :, 0:2].detach().numpy()
            ado_pos_wo = ado_traj_wo[i_ado, i_mode, :, 0:2].detach().numpy()

            axs[0].plot(ado_pos[:, 0], ado_pos[:, 1], "-*", color=ado_color, label=f"{ado_id}")
            _add_agent_representation(env.ado_ghosts[i].agent.state, ado_color, ado_id, ax=axs[0])
            axs[0].plot(ado_pos_wo[:, 0], ado_pos_wo[:, 1], "--", color=ado_color, label=f"{ado_id}_wo")

        optimization_step = log["iter_count"][k]
        axs[0].set_xlim(env.axes[0])
        axs[0].set_ylim(env.axes[1])
        axs[0].grid()
        axs[0].legend()
        axs[0].set_title(f"Optimization - Step {optimization_step} - Horizon {horizon}")

        # Plot agent velocities for resulting solution vs base-line ego trajectory for current optimization step.
        ado_velocity_norm = np.linalg.norm(ado_traj[:, :, :, 2:4].detach().numpy(), axis=3)
        ego_velocity_norm = np.linalg.norm(x4[:, 2:4].detach().numpy(), axis=1)
        for i in range(env.num_ado_ghosts):
            i_ado, i_mode = env.ghost_to_ado_index(i)
            ado_id, ado_color = env.ado_ghosts[i].id, env.ado_ghosts[i].agent.color
            axs[1].plot(time_axis, ado_velocity_norm[i_ado, i_mode, :], color=ado_color, label=f"{ado_id}_current")
        axs[1].plot(time_axis, ego_velocity_norm, color=env.ego.color, label="ego_current")
        axs[1].set_title("velocities [m/s]")
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
        axs[2].grid()
        axs[2].legend()

        # Plot ego control inputs (depending on ego agent type).
        ego_controls_norm = np.linalg.norm(env.ego.roll_trajectory(trajectory=x4, dt=env.dt).detach().numpy(), axis=1)
        axs[3].plot(time_axis[:-1], ego_controls_norm, color=env.ego.color)  # input have size = T - 1
        axs[3].set_title("control input")
        axs[3].grid()

        # Plot several parameter describing the optimization process, such as objective value, gradient and
        # the constraints (primal) infeasibility.
        for i, vis_key in enumerate(vis_keys):
            for name, data in log.items():
                if vis_key not in name:
                    continue
                axs[i + 4].plot(log["iter_count"][:k], np.log(np.asarray(data[:k]) + 1e-8), label=name)
            axs[i + 4].set_title(f"log_{vis_key}")
            axs[i + 4].legend()
            axs[i + 4].grid()

        return axs

    if not last_only:
        anim = FuncAnimation(fig, update, frames=len(log["iter_count"]), interval=200)
        anim.save(f"{file_path}.gif", dpi=60, writer='imagemagick')
    else:
        axs = update(len(log["iter_count"]) - 1)
        plt.savefig(f"{file_path}.png")
    logging.info(f"Optimisation visualisation stored in {file_path}")


# ##########################################################################
# Atomic plotting functions ################################################
# ##########################################################################
def _add_agent_representation(state: torch.Tensor, color: np.ndarray, name: Union[str, None], ax: plt.Axes):
    assert check_state(state, enforce_temporal=False), "state vector is invalid"
    state = state.detach().numpy()

    # Add circle for agent itself.
    ado_circle = plt.Circle(state[0:2], 0.2, color=color, clip_on=True)
    ax.add_artist(ado_circle)

    # Add agent id description.
    if id is not None:
        ax.text(state[0], state[1], name, fontsize=8)

    return ax
