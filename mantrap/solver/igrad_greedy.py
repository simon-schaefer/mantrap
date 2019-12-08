import logging

import numpy as np
import torch

from mantrap.agents import DoubleIntegratorDTAgent, IntegratorDTAgent
from mantrap.simulation.abstract import Simulation
from mantrap.solver.abstract import Solver


class IGradGreedySolver(Solver):
    """Idea:
    1) Find trajectory for ego from initial to goal state
    2) Use interactive gradients to minimize interference between ego and ados
    3) Transform to dynamical feasible trajectory
    Zhu, Schmerling, Pavone "A convex optimization approach to smooth trajectories for motion planning with car-like robots"
    """

    def __init__(self, sim: Simulation, goal: np.ndarray):
        super(IGradGreedySolver, self).__init__(sim, goal=goal)

    def _determine_ego_action(self, env: Simulation, k: int, traj_opt: np.ndarray) -> np.ndarray:
        path_base = np.vstack(
            (
                np.linspace(env.ego.state[0], self.goal[0], self._planning_horizon - k),
                np.linspace(env.ego.state[1], self.goal[1], self._planning_horizon - k),
            )
        ).T

        # Correct trajectory based on gradient of the sum of forces acting on the ados w.r.t. the ego base
        # trajectory. Based on the updated position, determine the velocity and update it.
        graph = env.build_graph_from_agents(ego_state=np.hstack((path_base[1, :], np.zeros(3))))
        ado_grads = np.zeros((env.num_ados, 2))
        for ia, ado in enumerate(env.ados):
            ado_force_norm = graph[f"{ado.id}_force_norm"]
            ado_grads[ia, :] = (
                torch.autograd.grad(ado_force_norm, graph["ego_position"], retain_graph=True)[0].detach().numpy()
            )
            logging.debug(f"solver @ k={k + 1} [ado_{ado.id}]: force={ado_force_norm.detach().numpy()}")
            logging.debug(f"solver @ k={k + 1} [ado_{ado.id}]: grad={ado_grads[ia, :]}")
        interactive_grad = np.mean(ado_grads, axis=0)

        # Update the position based on the force gradient (= acceleration gradient since m = 1kg) w.r.t. to
        # the position. TODO: Physically feasible dx -> dF/dx ??
        if env.ego.__class__ == IntegratorDTAgent:
            position_new = path_base[1, :] - interactive_grad * pow(env.dt, 2)
            ego_action = (position_new - env.ego.position) / env.dt
        elif env.ego.__class__ == DoubleIntegratorDTAgent:
            start_end_base = np.vstack(
                (
                    np.linspace(traj_opt[0, 0], self.goal[0], self._planning_horizon),
                    np.linspace(traj_opt[0, 1], self.goal[1], self._planning_horizon),
                )
            ).T
            tracking_force = start_end_base[k, :] - env.ego.position
            ego_action = -interactive_grad * pow(env.dt, 2) + 0.6 * tracking_force
        else:
            raise NotImplementedError(f"Ego type {env.ego.__class__} not supported !")
        return ego_action
