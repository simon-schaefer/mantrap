import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

import mantrap.constants
from mantrap.agents.agent import Agent
from mantrap.agents import DoubleIntegratorDTAgent
from mantrap.simulation.simulation import ForcesBasedSimulation


class DistanceFieldSimulation(ForcesBasedSimulation):
    def __init__(
        self,
        ego_type: Agent.__class__ = None,
        ego_kwargs: Dict[str, Any] = None,
        x_axis: Tuple[float, float] = mantrap.constants.sim_x_axis_default,
        y_axis: Tuple[float, float] = mantrap.constants.sim_y_axis_default,
        dt: float = mantrap.constants.sim_dt_default,
    ):
        super(DistanceFieldSimulation, self).__init__(ego_type, ego_kwargs, x_axis=x_axis, y_axis=y_axis, dt=dt)

    def add_ado(self, **ado_kwargs):
        """Add another DTV ado to the simulation. Since this simulation is based on forces the control input
        of each agent is a force, equivalent to some acceleration. Therefore it must be a double integrator. """
        super(DistanceFieldSimulation, self)._add_ado(DoubleIntegratorDTAgent, **ado_kwargs)

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
        --> Input = position of ados and ego state
        --> Output = Force acting on every ado"""
        assert len(ado_positions) == self.num_ados, "number of ado positions and internal number must match"

        # Define simulation parameters (as defined in the paper).
        num_ados = self.num_ados
        ado_ids = self.ado_ids
        sigma = mantrap.constants.sim_distance_field_sigma

        def _repulsive_force(alpha_position: torch.Tensor, beta_position: torch.Tensor):
            relative_distance = torch.sub(alpha_position, beta_position)
            norm_relative_distance = torch.max(torch.norm(relative_distance), torch.from_numpy(np.array([1e-10])))
            direction = torch.div(relative_distance, norm_relative_distance)
            return torch.exp(-norm_relative_distance / sigma) * direction

        # Graph initialization - Add ados and ego to graph (position only).
        graph = {}
        for i in range(num_ados):
            iid = ado_ids[i]
            graph[f"{iid}_position"] = ado_positions[i]
            if not is_intermediate:
                graph[f"{iid}_position"].requires_grad = True
        if ego_position is not None and ego_velocity is not None:
            graph["ego_position"] = ego_position
            if not is_intermediate:
                graph["ego_position"].requires_grad = True

        # Make graph with resulting force as an output.
        for i in range(num_ados):
            iid = ado_ids[i]
            graph[f"{iid}_force"] = torch.zeros(2)

            # Interactive force - Repulsive potential field by every other agent.
            for j in range(num_ados):
                jid = ado_ids[j]
                if iid == jid:
                    continue
                f_repulsive = _repulsive_force(graph[f"{iid}_position"], graph[f"{jid}_position"])
                interaction_force = torch.sub(graph[f"{iid}_force"], f_repulsive)
                graph[f"{iid}_force"] = interaction_force
                logging.debug(f"simulation [ado_{iid}]: interaction force ado {jid} -> {interaction_force.data}")

            # Interactive force w.r.t. ego - Repulsive potential field.
            if ego_position is not None and ego_velocity is not None:
                f_repulsive = _repulsive_force(graph[f"{iid}_position"], graph["ego_position"])
                interaction_force = torch.sub(graph[f"{iid}_force"], f_repulsive)
                graph[f"{iid}_force"] = interaction_force
                logging.debug(f"simulation [ado_{iid}]: interaction force ego -> {interaction_force.data}")

            # Summarize (standard) graph elements.
            graph[f"{iid}_force_norm"] = torch.norm(graph[f"{iid}_force"])

        # Graph health check.
        assert self.graph_check(graph=graph)

        return graph
