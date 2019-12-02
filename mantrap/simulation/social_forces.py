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
        fluctuations: float = mantrap.constants.sim_social_forces_fluctuations,
    ):
        super(SocialForcesSimulation, self).__init__(ego_type, ego_kwargs, x_axis=x_axis, y_axis=y_axis, dt=dt)
        self._ado_goals = []
        self._fluctuations = fluctuations

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
        :param fluctuations: additive noise (fluctuations) to resulting force on agent to model uncertainties and escape
                             local minimums
        :return: predicted trajectory for each ado (deterministic !).
        """
        if ego_trajectory is not None:
            assert ego_trajectory.shape[0] >= t_horizon, "t_horizon must match length of ego trajectory"

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
            ego_state = ego_trajectory[t, :] if ego_trajectory is not None else None
            graph_at_t = self._build_graph(ados_sim, ego_state=ego_state)

            # Evaluate graph.
            for i in range(num_ados):
                forces[t, i, :] = graph_at_t[f"{ados_sim[i].id}_force"].detach().numpy()

                forces[t, i, :] = forces[t, i, :] + np.random.rand(2) * self._fluctuations

                ados_sim[i].update(forces[t, i, :], self.dt)  # assuming m = 1 kg

        # Collect histories of simulated ados (last t_horizon steps are equal to future trajectories).
        trajectories = np.asarray([ado.history[-t_horizon:, :] for ado in ados_sim])
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

    def _build_graph(self, ados: List[Agent], ego_state: np.ndarray = None) -> Dict[str, torch.Tensor]:
        """Graph:
        --> Input = position & velocities of ados (and ego state & trajectory later on)
        --> Output = Force acting on every ado"""
        # Define simulation parameters (as defined in the paper).
        num_ados = len(ados)
        tau = 0.5  # [s] relaxation time (assumed to be uniform over all agents).
        v_0 = 2.1  # [m2s-2] repulsive field constant.
        sigma = 0.1  # [m] repulsive field exponent constant.

        graph = {}

        # Add ados to graph as an input - Properties such as goal, position and velocity.
        for i in range(num_ados):
            iid = ados[i].id
            graph[f"{iid}_goal"] = torch.tensor(self._ado_goals[i].astype(float))
            graph[f"{iid}_position"] = torch.tensor(ados[i].position.astype(float))
            graph[f"{iid}_position"].requires_grad = True
            graph[f"{iid}_velocity"] = torch.tensor(ados[i].velocity.astype(float))
            # graph[f"{iid}_velocity"].requires_grad = True

        # Make graph with resulting force as an output.
        for i in range(num_ados):
            iid = ados[i].id

            # Destination force - Force pulling the ado to its assigned goal position.
            direction = torch.sub(graph[f"{iid}_goal"], graph[f"{iid}_position"])
            direction = normalize_torch(direction)
            speed = torch.norm(graph[f"{iid}_velocity"])
            graph[f"{iid}_force"] = torch.sub(direction * speed, graph[f"{iid}_velocity"]) * 1 / tau

            def _repulsive_force(
                alpha_position: torch.Tensor,
                beta_position: torch.Tensor,
                alpha_velocity: torch.Tensor,
                beta_velocity: torch.Tensor,
            ):

                # Relative properties and their norms.
                relative_distance = torch.sub(alpha_position, beta_position)
                relative_distance.retain_grad()  # get gradient without being leaf node
                relative_velocity = torch.sub(alpha_velocity, beta_velocity)

                norm_relative_distance = torch.norm(relative_distance)
                norm_relative_velocity = torch.norm(relative_velocity)
                norm_diff_position = torch.sub(relative_distance, relative_velocity * self.dt).norm()

                # Alpha-Beta potential field.
                b1 = torch.add(norm_relative_distance, norm_diff_position)
                b2 = self.dt * norm_relative_velocity
                b = 0.5 * torch.sqrt(torch.sub(torch.pow(b1, 2), torch.pow(b2, 2)))
                v = v_0 * torch.exp(-b / sigma)

                # The repulsive force between agents is the negative gradient of the other (beta -> alpha)
                # potential field. Therefore subtract the gradient of V w.r.t. the relative distance.
                return torch.autograd.grad(v, relative_distance)[0]

            # Interactive force - Repulsive potential field by every other agent.
            for j in range(num_ados):
                jid = ados[j].id
                if iid == jid:
                    continue
                v_grad = _repulsive_force(
                    graph[f"{iid}_position"],
                    graph[f"{jid}_position"],
                    graph[f"{iid}_velocity"],
                    graph[f"{jid}_velocity"],
                )
                graph[f"{iid}_force"] = torch.sub(graph[f"{iid}_force"], v_grad)

            # Interactive force w.r.t. ego - Repulsive potential field.
            if ego_state is not None:
                graph[f"ego_position"] = torch.tensor(ego_state[0:2].astype(float))
                graph[f"ego_position"].requires_grad = True
                graph[f"ego_velocity"] = torch.tensor(ego_state[2:4].astype(float))
                v_grad = _repulsive_force(
                    graph[f"{iid}_position"], graph[f"ego_position"], graph[f"{iid}_velocity"], graph[f"ego_velocity"]
                )
                graph[f"{iid}_force"] = torch.sub(graph[f"{iid}_force"], v_grad)

        return graph
