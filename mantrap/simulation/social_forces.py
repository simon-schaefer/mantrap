from typing import Any, Dict, List, Tuple

import numpy as np
import torch

import mantrap.constants
from mantrap.utility.linalg import normalize_torch
from mantrap.agents.agent import Agent
from mantrap.agents import DoubleIntegratorDTAgent
from mantrap.simulation.abstract import ForcesBasedSimulation


class SocialForcesSimulation(ForcesBasedSimulation):
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

    def add_ado(self, goal_position: np.ndarray, **ado_kwargs):
        """Add another DTV ado to the simulation. In the social forces model every agent is assigned to some goal,
        so next to the ado itself this goal position has to be added."""
        assert goal_position.size == 2, "goal position must be two-dimensional (x, y)"

        super(SocialForcesSimulation, self)._add_ado(DoubleIntegratorDTAgent, **ado_kwargs)
        self._ado_goals.append(goal_position)

    def build_graph(
        self,
        ado_positions: List[torch.Tensor],
        ado_velocities: List[torch.Tensor],
        ego_position: torch.Tensor = None,
        ego_velocity: torch.Tensor = None,
        is_intermediate: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Graph:
        --> Input = position & velocities of ados and ego state
        --> Output = Force acting on every ado"""
        assert len(ado_positions) == self.num_ados, "number of ado positions and internal number must match"
        assert len(ado_velocities) == self.num_ados, "number of ado velocities and internal number must match"

        # Define simulation parameters (as defined in the paper).
        num_ados = self.num_ados
        ado_ids = self.ado_ids
        tau = 0.5  # [s] relaxation time (assumed to be uniform over all agents).
        v_0 = 2.1  # [m2s-2] repulsive field constant.
        sigma = 0.1  # [m] repulsive field exponent constant.

        # Repulsive force introduced by every other agent (depending on relative position and (!) velocity).
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
            return torch.autograd.grad(v, relative_distance, create_graph=True)[0]

        # Graph initialization - Add ados and ego to graph (position, velocity and goals).
        graph = {}
        for i in range(num_ados):
            iid = ado_ids[i]
            graph[f"{iid}_goal"] = torch.tensor(self._ado_goals[i].astype(float))
            graph[f"{iid}_position"] = ado_positions[i]
            graph[f"{iid}_velocity"] = ado_velocities[i]
            if not is_intermediate:
                graph[f"{iid}_position"].requires_grad = True
                # graph[f"{iid}_velocity"].requires_grad = True
        if ego_position is not None and ego_velocity is not None:
            graph["ego_position"] = ego_position
            graph["ego_velocity"] = ego_velocity
            if not is_intermediate:
                graph["ego_position"].requires_grad = True

        # Make graph with resulting force as an output.
        for i in range(num_ados):
            iid = ado_ids[i]

            # Destination force - Force pulling the ado to its assigned goal position.
            direction = torch.sub(graph[f"{iid}_goal"], graph[f"{iid}_position"])
            direction = normalize_torch(direction)
            speed = torch.norm(graph[f"{iid}_velocity"])
            graph[f"{iid}_force"] = torch.sub(direction * speed, graph[f"{iid}_velocity"]) * 1 / tau

            # Interactive force - Repulsive potential field by every other agent.
            for j in range(num_ados):
                jid = ado_ids[j]
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
            if ego_position is not None and ego_velocity is not None:
                v_grad = _repulsive_force(
                    graph[f"{iid}_position"], graph["ego_position"], graph[f"{iid}_velocity"], graph["ego_velocity"]
                )
                graph[f"{iid}_force"] = torch.sub(graph[f"{iid}_force"], v_grad)

            # Summarize (standard) graph elements.
            graph[f"{iid}_force_norm"] = torch.norm(graph[f"{iid}_force"])

        # Graph health check.
        assert self.graph_check(graph=graph)

        return graph
