from copy import deepcopy
import logging

import numpy as np
import torch

from mantrap.agents import DoubleIntegratorDTAgent, IntegratorDTAgent
from mantrap.simulation.abstract import Simulation
from mantrap.solver.abstract import Solver


class IGradPredictiveSolver(Solver):
    def __init__(self, sim: Simulation, goal: np.ndarray):
        super(IGradPredictiveSolver, self).__init__(sim, goal=goal)

    def _determine_ego_action(self, env: Simulation, k: int, traj_opt: np.ndarray) -> np.ndarray:
        prediction_horizon = min(20, self._planning_horizon - k)
        pred_env = deepcopy(env)
        graphs = []

        assert pred_env.ego.__class__ == DoubleIntegratorDTAgent, "currently only double integrator egos are supported"
        # Build first graph for the next state, in case no forces are applied on the ego.
        ado_traj, ego_next_state = pred_env.step(ego_policy=np.zeros(2))
        graphs.append(pred_env.build_graph_from_agents(ego_state=ego_next_state))
        # Build the graph iteratively for the whole prediction horizon.
        # ego_trajectory = pred_env.unroll_ego_trajectory(np.zeros((prediction_horizon, 2)))
        for kp in range(prediction_horizon):
            ado_positions, ado_velocities = [], []
            for ado in pred_env.ados:
                assert ado.__class__ == DoubleIntegratorDTAgent
                ado_force = graphs[-1][f"{ado.id}_force"]
                ado_velocity = graphs[-1][f"{ado.id}_velocity"]
                ado_position = graphs[-1][f"{ado.id}_position"]
                ado_vel_next = ado_velocity + pred_env.dt * ado_force
                ado_pos_next = ado_position + pred_env.dt * ado_velocity + 0.5 * pred_env.dt ** 2 * ado_force
                ado_positions.append(ado_pos_next)
                ado_velocities.append(ado_vel_next)
            ego_pos = graphs[-1]["ego_position"] + pred_env.dt * graphs[-1]["ego_velocity"]
            ego_vel = graphs[-1]["ego_velocity"]
            graph_kp = pred_env.build_graph(ado_positions, ado_velocities, ego_pos, ego_vel, is_intermediate=True)
            graphs.append(graph_kp)

        # Correct trajectory based on gradient of the sum of forces over multiple time-steps by determining the
        # gradient of the force acting on each ado w.r.t. the ego position in the first time-step.
        ado_grads = np.zeros((env.num_ados, 2))
        for ia, ado in enumerate(env.ados):
            ado_force_norm = graphs[-1][f"{ado.id}_force_norm"]
            ado_grads[ia, :] = (
                torch.autograd.grad(ado_force_norm, graphs[0]["ego_position"], retain_graph=True)[0].detach().numpy()
            )
            logging.debug(f"solver @ k={k + 1} [ado_{ado.id}]: force={ado_force_norm.detach().numpy()}")
            logging.debug(f"solver @ k={k + 1} [ado_{ado.id}]: grad={ado_grads[ia, :]}")
        interactive_grad = np.mean(ado_grads, axis=0)
        interactive_force = -interactive_grad

        # Update the position based on the force gradient (= acceleration gradient since m = 1kg) w.r.t. to
        # the position. TODO: Physically feasible dx -> dF/dx ??
        start_end_base = np.vstack(
            (
                np.linspace(traj_opt[0, 0], self.goal[0], self._planning_horizon),
                np.linspace(traj_opt[0, 1], self.goal[1], self._planning_horizon),
            )
        ).T
        tracking_force = start_end_base[k, :] - env.ego.position
        logging.debug(f"solver @ k={k + 1} [force]: interaction={interactive_grad}, tracking={tracking_force}")
        ego_action = 1e4 * interactive_force + 0.6 * tracking_force
        return ego_action
