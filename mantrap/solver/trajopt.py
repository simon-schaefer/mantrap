from copy import deepcopy
import logging
from typing import Tuple, Union

import numpy as np
import torch

from mantrap.constants import planning_horizon_default
from mantrap.agents import DoubleIntegratorDTAgent, IntegratorDTAgent
from mantrap.simulation.abstract import Simulation
from mantrap.solver.abstract import Solver


class TrajOptSolver(Solver):
    """Idea:
    1) Find trajectory for ego from initial to goal state
    2) Use interactive gradients to minimize interference between ego and ados
    3) Transform to dynamical feasible trajectory
    Zhu, Schmerling, Pavone "A convex optimization approach to smooth trajectories for motion planning with car-like robots"
    """

    def __init__(self, sim: Simulation, goal: np.ndarray):
        super(TrajOptSolver, self).__init__(sim, goal=goal)

    def solve(self, planning_horizon: int = planning_horizon_default) -> Tuple[Union[np.ndarray, None], np.ndarray]:
        solver_env = deepcopy(self._env)
        traj_opt = np.zeros((planning_horizon, 6))
        traj_opt[0, :] = np.hstack((solver_env.ego.state, solver_env.sim_time))

        ado_trajectories = np.zeros((solver_env.num_ados, solver_env.num_ado_modes, planning_horizon, 6))
        for ia, ado in enumerate(solver_env.ados):
            ado_trajectories[ia, :, 0, :] = np.hstack((ado.state, solver_env.sim_time))

        logging.debug(f"Starting trajectory optimization solving for {planning_horizon} steps ...")
        for t in range(planning_horizon - 1):
            path_base = np.vstack(
                (
                    np.linspace(traj_opt[t, 0], self.goal[0], planning_horizon - t),
                    np.linspace(traj_opt[t, 1], self.goal[1], planning_horizon - t),
                )
            ).T

            # Correct trajectory based on gradient of the sum of forces acting on the ados w.r.t. the ego base
            # trajectory. Based on the updated position, determine the velocity and update it.
            graph = solver_env.build_graph(ego_state=np.hstack((path_base[1, :], np.zeros(3))))
            ado_grads = np.zeros((solver_env.num_ados, 2))
            for ia, ado in enumerate(solver_env.ados):
                ado_force_norm = graph[f"{ado.id}_force_norm"]
                ado_grads[ia, :] = (
                    torch.autograd.grad(ado_force_norm, graph["ego_position"], retain_graph=True)[0].detach().numpy()
                )
                logging.debug(f"solver @ t={t+1} [ado_{ado.id}]: force={ado_force_norm.detach().numpy()}")
                logging.debug(f"solver @ t={t+1} [ado_{ado.id}]: grad={ado_grads[ia, :]}")
            interactive_grad = np.mean(ado_grads, axis=0)

            # Update the position based on the force gradient (= acceleration gradient since m = 1kg) w.r.t. to
            # the position. TODO: Physically feasible dx -> dF/dx ??
            if solver_env.ego.__class__ == IntegratorDTAgent:
                position_new = path_base[1, :] - interactive_grad * pow(solver_env.dt, 2)
                position_old = traj_opt[t, 0:2]
                ego_action = (position_new - position_old) / solver_env.dt
            elif solver_env.ego.__class__ == DoubleIntegratorDTAgent:
                start_end_base = np.vstack(
                    (
                        np.linspace(traj_opt[0, 0], self.goal[0], planning_horizon),
                        np.linspace(traj_opt[0, 1], self.goal[1], planning_horizon),
                    )
                ).T
                tracking_force = start_end_base[t, :] - traj_opt[t, 0:2]
                ego_action = - interactive_grad * pow(solver_env.dt, 2) + 0.1 * tracking_force
            else:
                raise NotImplementedError(f"Ego type {solver_env.ego.__class__} not supported !")

            # Forward simulate environment.
            ado_traj, ego_state = solver_env.step(ego_policy=ego_action)
            ado_trajectories[:, :, t + 1, :] = ado_traj[:, :, 0, :]
            traj_opt[t + 1, :] = ego_state

            logging.debug(f"solver @ t={t+1}: interaction_force = {interactive_grad}")

        logging.debug(f"Finishing up trajectory optimization solving")
        return traj_opt, ado_trajectories
