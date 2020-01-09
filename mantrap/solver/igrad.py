from abc import abstractmethod
from copy import deepcopy
import logging

import numpy as np
import torch

import mantrap.constants
from mantrap.agents import IntegratorDTAgent, DoubleIntegratorDTAgent
from mantrap.simulation.simulation import Simulation
from mantrap.solver.solver import Solver


class IGradSolver(Solver):
    """The basic idea of the IGrad solver family is to perturb some nominal trajectory from the robot initial to its
    goal state so that the interaction with the other agents in the scene is minimized. Therefore for every ado in the
    scene the simulation graph's gradient of the ado's trajectory w.r.t. the ego's next state is determined, in order
    to "push" the next ego state in an interaction-minimizing direction. The perturbation vector rho_ego(x, t) then
    is the following:

    rho_ego(x, t) = - alpha * sum_ados( grad_x_ego(x_ado(t + n) )

    Currently, just a straight line from initial to goal position is used as a nominal trajectory.
    """

    def determine_ego_action(self, env: Simulation) -> np.ndarray:
        # Next ego position should be some distance ahead in goal direction. The distance is thereby dependent on the
        # remaining distance between the current ego state and the goal state as well as its velocity.
        goal_distance = np.linalg.norm(self.goal - env.ego.position)
        goal_direction = self.goal - env.ego.position / goal_distance
        ego_pos_next = goal_direction * min(np.linalg.norm(env.ego.velocity) * env.dt, goal_distance)

        # Correct trajectory based on gradient of the sum of forces acting on the ados w.r.t. the ego base
        # trajectory. Based on the updated position, determine the velocity and update it.
        interaction_grad = self.compute_interaction_grad(env=env, ego_query=ego_pos_next)
        logging.info(f"solver: interaction_grad = {interaction_grad}")

        # Update the position based on the force gradient (= acceleration gradient since m = 1kg) w.r.t. to
        # the position. TODO: Physically feasible dx -> dF/dx ??
        if env.ego.__class__ == IntegratorDTAgent:
            position_new = ego_pos_next - 100 * interaction_grad * pow(env.dt, 2)
            ego_action = (position_new - env.ego.position) / env.dt
        # elif env.ego.__class__ == DoubleIntegratorDTAgent:
        # ego_base = traj_opt[0, 0:2] + k / self._planning_horizon * (self.goal - traj_opt[0, 0:2])
        # tracking_force = ego_base - env.ego.position
        # logging.info(f"solver: tracking force = {tracking_force}")
        # ego_action = self._alpha.dot(np.array([-interaction_grad, tracking_force]))
        else:
            raise NotImplementedError(f"Ego type {env.ego.__class__} not supported !")
        return ego_action

    @abstractmethod
    def compute_interaction_grad(self, env: Simulation, ego_query: np.ndarray):
        pass

    @staticmethod
    def close_ados(env: Simulation):
        ados_close = []
        for ado in env.ados:
            ego_ado_distance = np.linalg.norm(env.ego.position - ado.position)
            if ego_ado_distance < mantrap.constants.igrad_radius:
                ados_close.append(ado)
        return ados_close


class IGradGreedySolver(IGradSolver):
    def compute_interaction_grad(self, env: Simulation, ego_query: np.ndarray):
        graph = env.build_graph_from_agents(ego_state=np.hstack((ego_query, np.zeros(3))))

        ados_close = self.close_ados(env=env)
        if len(ados_close) == 0:
            return np.zeros(2)

        ado_grads = np.zeros((len(ados_close), 2))
        for ia, ado in enumerate(ados_close):
            ado_force_norm = graph[f"{ado.id}_force_norm"]
            ado_grads[ia, :] = (
                torch.autograd.grad(ado_force_norm, graph["ego_position"], retain_graph=True)[0].detach().numpy()
            )
            logging.info(f"solver [ado_{ado.id}]: force={ado_force_norm.detach().numpy()}")
            logging.info(f"solver [ado_{ado.id}]: grad={ado_grads[ia, :]}")
        return np.mean(ado_grads, axis=0)


class IGradPredictiveSolver(IGradSolver):
    def compute_interaction_grad(self, env: Simulation, ego_query: np.ndarray):
        pred_env = deepcopy(env)
        graphs = []

        assert pred_env.ego.__class__ == IntegratorDTAgent, "currently only single integrator egos are supported"
        # Build first graph for the next state, in case no forces are applied on the ego.
        ego_policy = np.zeros(2)  # pred_env.ego.velocity
        ado_traj, ego_next_state = pred_env.step(ego_policy=ego_policy)
        graphs.append(pred_env.build_graph_from_agents(ego_state=ego_next_state))
        # Build the graph iteratively for the whole prediction horizon.
        prediction_horizon = mantrap.constants.igrad_predictive_horizon
        for kp in range(prediction_horizon):
            logging.debug(f"solver: building graph over time - step kp = {kp}")

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

            # The ego movement is, of cause, unknown, since we try to find it here. The best guess we can do is
            # that the ego does not change its trajectory, by keeping its linear velocity (constant-velocity
            # movement). Even if it does not perfectly reflect the actual movement of the ego, it probably is close
            # to the actual trajectory, within the (comparably short) prediction horizon.
            ego_pos = graphs[-1]["ego_position"] + pred_env.dt * graphs[-1]["ego_velocity"]
            ego_vel = graphs[-1]["ego_velocity"]

            graph_kp = pred_env.build_graph(ado_positions, ado_velocities, ego_pos, ego_vel, is_intermediate=True)
            graphs.append(graph_kp)
        logging.debug("solver: finished building graph")

        ados_close = self.close_ados(env=env)
        if len(ados_close) == 0:
            return np.zeros(2)

        # Correct trajectory based on gradient of the sum of forces over multiple time-steps by determining the
        # gradient of the force acting on each ado w.r.t. the ego position in the first time-step.
        ado_grads = np.zeros((len(ados_close), 2))
        for ia, ado in enumerate(ados_close):
            ado_force_norm = graphs[-1][f"{ado.id}_force_norm"]
            ado_grads[ia, :] = (
                torch.autograd.grad(ado_force_norm, graphs[0]["ego_position"], retain_graph=True)[0].detach().numpy()
            )
            logging.info(f"solver [ado_{ado.id}]: force={ado_force_norm.detach().numpy()}, grad={ado_grads[ia, :]}")
        return np.mean(ado_grads, axis=0)
