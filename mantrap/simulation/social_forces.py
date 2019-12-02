import copy
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

import mantrap.constants
from mantrap.utility.linalg import normalize_torch
from mantrap.agents.agent import Agent
from mantrap.agents import DoubleIntegratorDTAgent
from mantrap.simulation.abstract import Simulation


class SocialForcesSimulation(Simulation):
    def __init__(
        self,
        ego_type: Agent.__class__ = None,
        ego_kwargs: Dict[str, Any] = None,
        x_axis: Tuple[float, float] = mantrap.constants.sim_x_axis_default,
        y_axis: Tuple[float, float] = mantrap.constants.sim_y_axis_default,
        dt: float = mantrap.constants.sim_dt_default,
    ):
        super(SocialForcesSimulation, self).__init__(ego_type, ego_kwargs, x_axis=x_axis, y_axis=y_axis, dt=dt)
        self._ado_goals = []

    def predict(
        self,
        t_horizon: int = mantrap.constants.t_horizon_default,
        ego_trajectory: np.ndarray = None,
        return_policies: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Compute the trajectory for each ado in the environment.
        Therefore apply the social forces model described in "Social force model for pedestrian dynamics" (Helbing),
        only taking the destination forces and interactive forces into account. The social forces model is
        deterministic (assuming the initial position and dynamics of the agents are deterministic). Hence, for each
        ado merely one trajectory is predicted.

        :param t_horizon: prediction horizon (number of time-steps of length dt).
        :param ego_trajectory: planned ego trajectory (in case of dependence in behaviour between ado and ego).
        :param return_policies: return the actual system inputs (at every time to get trajectory).
        :return: predicted trajectory for each ado (deterministic !).
        """
        num_ados = len(self._ados)
        forces = np.zeros((t_horizon, num_ados, 2))

        # The social forces model predicts from one time-step to another, therefore the ados are actually updated in
        # each time step, in order to predict the next time-step. To not change the initial state, hence, the ados
        # vector is copied.
        ados_sim = copy.deepcopy(self.ados)
        for t in range(t_horizon):
            # Build graph based on simulation ados. Build it once and update it every time is surprisingly difficult
            # since the gradients/computations are not updated when one of the leafs is updated, resulting in the
            # same output. However the computational effort of building the graph is negligible (about 1 ms for
            # 2 agents on Mac Pro 2018).
            graph_at_t = self._build_graph(ados_sim)

            # Evaluate graph.
            for i in range(num_ados):
                forces[t, i, :] = graph_at_t[f"{ados_sim[i].id}_force"].detach().numpy()
                ados_sim[i].update(forces[t, i, :], self.dt)  # assuming m = 1 kg

        # Collect histories of simulated ados (last t_horizon steps are equal to future trajectories).
        trajectories = np.asarray([ado.history for ado in ados_sim])
        assert trajectories.shape[0] == num_ados, "each ado must be assigned to trajectory"

        if not return_policies:
            return trajectories
        else:
            return trajectories, forces

    def add_ado(self, goal_position: np.ndarray, **ado_kwargs):
        """Add another DTV ado to the simulation. In the social forces model every agent is assigned to some goal,
        so next to the ado itself this goal position has to be added."""
        assert goal_position.size == 2, "goal position must be two-dimensional (x, y)"

        super(SocialForcesSimulation, self)._add_ado(DoubleIntegratorDTAgent, **ado_kwargs)
        self._ado_goals.append(goal_position)

    def _build_graph(self, ados: List[Agent]) -> Dict[str, torch.Tensor]:
        """Graph:
        --> Input = position & velocities of ados (and ego state & trajectory later on)
        --> Output = Force acting on every ado"""
        # Define simulation parameters (as defined in the paper).
        num_ados = len(ados)
        tau = 0.5  # [s] relaxation time (assumed to be uniform over all agents).
        v_0 = 2.1  # [m2s-2] repulsive field constant.
        sigma = 0.3  # [m] repulsive field exponent constant.

        graph = {}

        # Add ados to graph as an input - Properties such as goal, position and velocity.
        for i in range(num_ados):
            iid = ados[i].id
            graph[f"{iid}_goal"] = torch.tensor(self._ado_goals[i].astype(float))
            graph[f"{iid}_position"] = torch.tensor(ados[i].position.astype(float))
            graph[f"{iid}_velocity"] = torch.tensor(ados[i].velocity.astype(float))

        # Make graph with resulting force as an output.
        for i in range(num_ados):
            iid = ados[i].id

            # Destination force - Force pulling the ado to its assigned goal position.
            direction = torch.sub(graph[f"{iid}_goal"], graph[f"{iid}_position"])
            direction = normalize_torch(direction)
            speed = torch.norm(graph[f"{iid}_velocity"])
            graph[f"{iid}_force"] = torch.sub(direction * speed, graph[f"{iid}_velocity"]) * 1 / tau

            # Interactive force - Repulsive potential field by every other agent.
            for j in range(num_ados):
                jid = ados[j].id
                if iid == jid:
                    continue

                # Relative properties and their norms.
                relative_distance = torch.sub(graph[f"{iid}_position"], graph[f"{jid}_position"])
                relative_distance.requires_grad = True
                relative_velocity = torch.sub(graph[f"{iid}_velocity"], graph[f"{jid}_velocity"])

                norm_relative_distance = torch.norm(relative_distance)
                norm_relative_velocity = torch.norm(relative_velocity)
                norm_diff_position = torch.sub(relative_distance, relative_velocity * self.dt).norm()

                # Alpha-Beta potential field.
                b1 = torch.add(norm_relative_distance, norm_diff_position)
                b2 = self.dt * norm_relative_velocity
                b = 0.5 * torch.sqrt(torch.sub(torch.pow(b1, 2), torch.pow(b2, 2)))
                v = v_0 * torch.exp(-b / sigma)
                v.backward()

                # The repulsive force between agents is the negative gradient of the other (beta -> alpha)
                # potential field. Therefore subtract the gradient of V w.r.t. the relative distance.
                graph[f"{iid}_force"] = torch.sub(graph[f"{iid}_force"], relative_distance.grad)

        return graph
