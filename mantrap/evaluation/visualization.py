from typing import Dict, List, Union

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import torch

from mantrap.environment.environment import GraphBasedEnvironment
from mantrap.utility.maths import Derivative2
from mantrap.utility.shaping import check_ego_trajectory, check_ado_trajectories, check_ego_state


def visualize(
    ego_planned: torch.Tensor,
    ado_planned: torch.Tensor,
    env: GraphBasedEnvironment,
    obj_dict: Dict[str, List[torch.Tensor]],
    inf_dict: Dict[str, List[torch.Tensor]],
    file_path: str,
    ego_trials: List[List[torch.Tensor]] = None,
    single_opt: bool = False,
):
    assert len(ego_planned.shape) == 3
    num_solver_calls = ego_planned.shape[0]
    assert len(ado_planned.shape) == 5 and ado_planned.shape[0] == num_solver_calls
    assert all([len(x) == num_solver_calls for x in obj_dict.values()])
    assert all([len(x) == num_solver_calls for x in inf_dict.values()])

    fig, ax = plt.subplots(figsize=(15, 15), constrained_layout=True)
    grid = plt.GridSpec(2 + 3 + 2, 2, wspace=0.4, hspace=0.3, figure=fig)
    plt.axis("off")

    axs = list()
    # Trajectory plot.
    axs.append(fig.add_subplot(grid[:3, :]))
    # Velocity  plot.
    axs.append(fig.add_subplot(grid[3 + 0, :]))
    # Acceleration plot.
    axs.append(fig.add_subplot(grid[3 + 1, :]))
    # Control input plot.
    axs.append(fig.add_subplot(grid[3 + 2, :]))
    # Objective & Constraint plot.
    axs.append(fig.add_subplot(grid[3 + 3, 0]))
    axs.append(fig.add_subplot(grid[3 + 3, 1]))

    def update(k):
        ego_trajectory = ego_planned[:, 0, :].detach()
        ego_planned_k = ego_planned[k].detach()
        ado_trajectory = ado_planned[:, :, :, 0, :].permute(1, 2, 0, 3).detach()
        ado_planned_k = ado_planned[k].detach()
        ego_trials_k = ego_trials[k] if ego_trials is not None else None
        k_ = k

        if single_opt:
            ego_trajectory = ego_planned[k].detach()
            ado_trajectory = ado_planned[k].detach()
            k_ = None

        plt.axis("off")
        for i in range(len(axs)):
            axs[i].cla()

        axs[0] = draw_trajectories(ego_planned_k, ado_planned_k, env=env, ego_traj_trials=ego_trials_k, ax=axs[0])
        axs[1] = draw_velocities(ado_trajectory, ego_traj=ego_trajectory, env=env, k=k_, ax=axs[1])
        axs[2] = draw_ado_accelerations(ado_trajectory, env=env, k=k_, ax=axs[2])
        axs[3] = draw_ego_controls(ego_trajectory, env=env, k=k_, ax=axs[3])

        for key, values in obj_dict.items():
            data = values[k] if not single_opt else values[:k]
            axs[4] = draw_values(data, label=key, env=env, ax=axs[4], k=k_)
        for key, values in inf_dict.items():
            data = values[k] if not single_opt else values[:k]
            axs[5] = draw_values(data, label=key, env=env, ax=axs[5], k=k_)

        axs[0].set_title(f"step {k}")
        return axs

    anim = FuncAnimation(fig, update, frames=num_solver_calls - 1, interval=300)
    anim.save(f"{file_path}.gif", dpi=60, writer='imagemagick')


# ##########################################################################
# Atomic plotting functions ################################################
# ##########################################################################
def draw_trajectories(
    ego_traj: torch.Tensor,
    ado_traj: torch.Tensor,
    env: GraphBasedEnvironment,
    ax: plt.Axes,
    ego_traj_trials: List[torch.Tensor] = None
):
    """Plot current and base solution in the scene. This includes the determined ego trajectory (x) as well as the
    resulting ado trajectories based on some environment."""

    def draw_agent_representation(state: torch.Tensor, color: np.ndarray, name: Union[str, None]):
        """Add circle for agent and agent id description."""
        assert check_ego_state(state, enforce_temporal=False), "state vector is invalid"
        state = state.detach().numpy()
        ado_circle = plt.Circle(state[0:2], 0.2, color=color, clip_on=True)
        ax.add_artist(ado_circle)
        if id is not None:
            ax.text(state[0], state[1], name, fontsize=8)

    assert check_ego_trajectory(ego_traj, pos_and_vel_only=True)
    assert check_ado_trajectories(ado_traj, ados=env.num_ados, pos_and_vel_only=True, t_horizon=ego_traj.shape[0])

    ego_trajectory_np = ego_traj.detach().numpy()
    ax.plot(ego_trajectory_np[:, 0], ego_trajectory_np[:, 1], "-", color=env.ego.color, label="ego")
    draw_agent_representation(ego_traj[0, :], env.ego.color, "ego")

    # Plot trial trajectories during optimisation process.
    if ego_traj_trials is not None:
        for ego_traj_trial in ego_traj_trials:
            ego_traj_trial_np = ego_traj_trial.detach().numpy()
            ax.plot(ego_traj_trial_np[:, 0], ego_traj_trial_np[:, 1], "--", color=env.ego.color, alpha=0.08)

    # Plot current and base resulting simulated ado trajectories in the scene.
    vis_env = env.copy()
    vis_env.step_reset(ego_state_next=None, ado_states_next=ado_traj[:, 0, 0, :])
    ado_traj_wo = vis_env.predict_wo_ego(t_horizon=ego_traj.shape[0])

    for ghost in env.ghosts:
        i_ado, i_mode = env.convert_ghost_id(ghost_id=ghost.id)
        ado_id, ado_color = ghost.id, ghost.agent.color
        ado_pos = ado_traj[i_ado, i_mode, :, 0:2].detach().numpy()
        ado_pos_wo = ado_traj_wo[i_ado, i_mode, :, 0:2].detach().numpy()

        ax.plot(ado_pos[:, 0], ado_pos[:, 1], "-*", color=ado_color, label=f"{ado_id}")
        draw_agent_representation(ado_traj[i_ado, i_mode, 0, :], ado_color, ado_id)
        ax.plot(ado_pos_wo[:, 0], ado_pos_wo[:, 1], "--", color=ado_color, label=f"{ado_id}_wo")

    ax.set_xlim(*env.axes[0])
    ax.set_ylim(*env.axes[1])
    ax.grid()
    ax.legend()
    return ax


def draw_velocities(ado_traj: torch.Tensor, ego_traj: torch.Tensor, env, ax: plt.Axes, k: Union[int, None] = 0):
    """Plot agent velocities for resulting solution vs base-line ego trajectory for current optimization step."""
    assert check_ego_trajectory(ego_traj, pos_and_vel_only=True)
    assert check_ado_trajectories(ado_traj, ados=env.num_ados, pos_and_vel_only=True, t_horizon=ego_traj.shape[0])
    time_axis = np.linspace(env.time, env.time + ego_traj.shape[0] * env.dt, num=ego_traj.shape[0])
    ado_velocity_norm = np.linalg.norm(ado_traj[:, :, :, 2:4].detach().numpy(), axis=3)
    ego_velocity_norm = np.linalg.norm(ego_traj[:, 2:4].detach().numpy(), axis=1)
    for ghost in env.ghosts:
        i_ado, i_mode = env.convert_ghost_id(ghost_id=ghost.id)
        ado_id, ado_color = ghost.id, ghost.agent.color
        ax.plot(time_axis, ado_velocity_norm[i_ado, i_mode, :], color=ado_color, label=f"{ado_id}_current")
    ax.plot(time_axis, ego_velocity_norm, color=env.ego.color, label="ego_current")
    if k is not None:
        ax.axvline(x=time_axis[k], color=np.array([1, 0, 0]))
    ax.set_title("velocities [m/s]")
    ax.grid()
    ax.legend()
    return ax


def draw_ado_accelerations(ado_traj: torch.Tensor, env: GraphBasedEnvironment, ax: plt.Axes, k: Union[int, None] = 0):
    """Plot agent accelerations for resulting solution vs base-line ego trajectory for current optimization step."""
    assert check_ado_trajectories(ado_traj, ados=env.num_ados, pos_and_vel_only=True)
    time_axis = np.linspace(env.time, env.time + ado_traj.shape[2] * env.dt, num=ado_traj.shape[2])
    dd = Derivative2(horizon=ado_traj.shape[2], dt=env.dt, velocity=True)
    ado_acceleration_norm = np.linalg.norm(dd.compute(ado_traj[:, :, :, 2:4]).detach().numpy(), axis=3)
    for ghost in env.ghosts:
        i_ado, i_mode = env.convert_ghost_id(ghost_id=ghost.id)
        ado_id, ado_color = ghost.id, ghost.agent.color
        ax.plot(time_axis, ado_acceleration_norm[i_ado, i_mode, :], color=ado_color, label=f"{ado_id}_current")
    if k is not None:
        ax.axvline(x=time_axis[k], color=np.array([1, 0, 0]))
    ax.set_title("accelerations [m/s^2]")
    ax.grid()
    ax.legend()
    return ax


def draw_ego_controls(ego_traj: torch.Tensor, env: GraphBasedEnvironment, ax: plt.Axes, k: Union[int, None] = 0):
    """Plot ego control inputs (depending on ego agent type)."""
    assert check_ego_trajectory(ego_traj, pos_and_vel_only=True)
    time_axis = np.linspace(env.time, env.time + ego_traj.shape[0] * env.dt, num=ego_traj.shape[0])
    ego_controls_norm = np.linalg.norm(env.ego.roll_trajectory(trajectory=ego_traj, dt=env.dt).detach().numpy(), axis=1)
    ego_controls_norm = np.concatenate((ego_controls_norm, np.zeros(1)))  # controls = T - 1 so stretch them
    ax.plot(time_axis, ego_controls_norm, color=env.ego.color)
    if k is not None:
        ax.axvline(x=time_axis[k], color=np.array([1, 0, 0]))
    ax.set_title("control input")
    ax.grid()
    return ax


def draw_values(values: torch.Tensor, label: str, env, ax: plt.Axes, k: Union[int, None] = 0):
    assert len(values.shape) == 1  # one-dimensional vector (!)
    time_axis = np.linspace(env.time, env.time + values.numel() * env.dt, num=values.numel())
    ax.plot(time_axis, np.log(np.asarray(values) + 1e-8), label=label)
    if k is not None:
        ax.axvline(x=time_axis[k], color=np.array([1, 0, 0]))
    ax.legend()
    ax.grid()
    return ax
