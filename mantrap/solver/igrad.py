from abc import abstractmethod
from copy import deepcopy
import logging

import numpy as np
import torch

from mantrap.agents import IntegratorDTAgent, DoubleIntegratorDTAgent
from mantrap.simulation.abstract import Simulation
from mantrap.solver.abstract import Solver


class IGradSolver(Solver):
    def __init__(self, sim: Simulation, goal: np.ndarray, alpha: np.ndarray):
        super(IGradSolver, self).__init__(sim, goal=goal)
        assert alpha.size == 2, "alpha vector must contain scaling parameters for derived action"
        self._alpha = alpha

    def _determine_ego_action(self, env: Simulation, k: int, traj_opt: np.ndarray) -> np.ndarray:
        logging.debug(f"solver @ time-step k = {k}")
        ego_next = env.ego.position + 1 / (self._planning_horizon - k) * (self.goal - env.ego.position)

        # Correct trajectory based on gradient of the sum of forces acting on the ados w.r.t. the ego base
        # trajectory. Based on the updated position, determine the velocity and update it.
        interaction_grad = self.compute_interaction_grad(env=env, ego_query=ego_next)
        logging.debug(f"solver: interaction_grad = {interaction_grad}")

        # Update the position based on the force gradient (= acceleration gradient since m = 1kg) w.r.t. to
        # the position. TODO: Physically feasible dx -> dF/dx ??
        if env.ego.__class__ == IntegratorDTAgent:
            position_new = ego_next - interaction_grad * env.dt
            ego_action = (position_new - env.ego.position) / env.dt
        elif env.ego.__class__ == DoubleIntegratorDTAgent:
            ego_base = traj_opt[0, 0:2] + k / self._planning_horizon * (self.goal - traj_opt[0, 0:2])
            tracking_force = ego_base - env.ego.position
            logging.debug(f"solver: tracking force = {tracking_force}")
            ego_action = self._alpha.dot(np.array([-interaction_grad, tracking_force]))
        else:
            raise NotImplementedError(f"Ego type {env.ego.__class__} not supported !")
        return ego_action

    @abstractmethod
    def compute_interaction_grad(self, env: Simulation, ego_query: np.ndarray):
        pass


class IGradGreedySolver(IGradSolver):
    def __init__(
        self, sim: Simulation, goal: np.ndarray, alpha: np.ndarray = np.array([1.0, 0.6]), radius: float = 3.0
    ):
        super(IGradGreedySolver, self).__init__(sim, goal, alpha=alpha)
        self._radius = radius

    def compute_interaction_grad(self, env: Simulation, ego_query: np.ndarray):
        graph = env.build_graph_from_agents(ego_state=np.hstack((ego_query, np.zeros(3))))

        ados_close = []
        for ado in env.ados:
            ego_ado_distance = np.linalg.norm(env.ego.position - ado.position)
            if ego_ado_distance < self._radius:
                ados_close.append(ado)
        if len(ados_close) == 0:
            return np.zeros(2)

        ado_grads = np.zeros((len(ados_close), 2))
        for ia, ado in enumerate(ados_close):
            ado_force_norm = graph[f"{ado.id}_force_norm"]
            ado_grads[ia, :] = (
                torch.autograd.grad(ado_force_norm, graph["ego_position"], retain_graph=True)[0].detach().numpy()
            )
            logging.debug(f"solver [ado_{ado.id}]: force={ado_force_norm.detach().numpy()}")
            logging.debug(f"solver [ado_{ado.id}]: grad={ado_grads[ia, :]}")
        return np.mean(ado_grads, axis=0)


class IGradPredictiveSolver(IGradSolver):
    def __init__(
        self, sim: Simulation, goal: np.ndarray, alpha: np.ndarray = np.array([1e4, 0.6]), prediction_horizon: int = 10
    ):
        super(IGradPredictiveSolver, self).__init__(sim, goal, alpha=alpha)
        self._prediction_horizon = prediction_horizon

    def compute_interaction_grad(self, env: Simulation, ego_query: np.ndarray):
        pred_env = deepcopy(env)
        graphs = []

        assert pred_env.ego.__class__ == DoubleIntegratorDTAgent, "currently only double integrator egos are supported"
        # Build first graph for the next state, in case no forces are applied on the ego.
        ado_traj, ego_next_state = pred_env.step(ego_policy=np.zeros(2))
        graphs.append(pred_env.build_graph_from_agents(ego_state=ego_next_state))
        # Build the graph iteratively for the whole prediction horizon.
        # ego_trajectory = pred_env.unroll_ego_trajectory(np.zeros((prediction_horizon, 2)))
        for kp in range(self._prediction_horizon):
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
            logging.debug(f"solver [ado_{ado.id}]: force={ado_force_norm.detach().numpy()}")
            logging.debug(f"solver [ado_{ado.id}]: grad={ado_grads[ia, :]}")
        return np.mean(ado_grads, axis=0)
