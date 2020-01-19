import copy
import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from mantrap.agents.agent import Agent
from mantrap.agents import DoubleIntegratorDTAgent
from mantrap.constants import (
    sim_x_axis_default,
    sim_y_axis_default,
    sim_dt_default,
    sim_social_forces_tau,
    sim_social_forces_v_0,
    sim_social_forces_sigma,
    sim_social_forces_min_goal_distance,
    sim_social_forces_max_interaction_distance,
)
from mantrap.utility.shaping import check_ado_trajectories
from mantrap.simulation.simulation import Simulation


class SocialForcesSimulation(Simulation):
    def __init__(
        self,
        ego_type: Agent.__class__ = None,
        ego_kwargs: Dict[str, Any] = None,
        x_axis: Tuple[float, float] = sim_x_axis_default,
        y_axis: Tuple[float, float] = sim_y_axis_default,
        dt: float = sim_dt_default,
        fluctuations: float = 0.0,
    ):
        super(SocialForcesSimulation, self).__init__(ego_type, ego_kwargs, x_axis=x_axis, y_axis=y_axis, dt=dt)
        self._ado_goals = []
        self._fluctuations = fluctuations

    def predict(
            self, t_horizon: int, ego_trajectory: np.ndarray = None, return_policies: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if ego_trajectory is not None:
            assert ego_trajectory.shape[0] >= t_horizon, "t_horizon must match length of ego trajectory"

        forces = np.zeros((t_horizon, self.num_ados, 2))

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
            graph_at_t = self.build_graph_from_agents(ados_sim, ego_state=ego_state)

            # Evaluate graph.
            for i in range(self.num_ados):
                forces[t, i, :] = graph_at_t[f"{ados_sim[i].id}_force"].detach().numpy()
                ados_sim[i].update(forces[t, i, :], dt=self.dt)  # assuming m = 1 kg

        # Collect histories of simulated ados (last t_horizon steps are equal to future trajectories).
        trajectories = np.expand_dims(np.asarray([ado.history[-t_horizon:, :] for ado in ados_sim]), axis=1)
        assert check_ado_trajectories(trajectories, t_horizon=t_horizon, num_ados=self.num_ados, num_modes=1)
        return trajectories if not return_policies else (trajectories, forces)

    def add_ado(self, **ado_kwargs):
        """Add another DTV ado to the simulation. In the social forces model every agent is assigned to some goal,
        so next to the ado itself this goal position has to be added."""
        assert "goal" in ado_kwargs.keys() and type(ado_kwargs["goal"]) == np.ndarray, "goal position required"
        assert ado_kwargs["goal"].size == 2, "goal position must be two-dimensional (x, y)"
        super(SocialForcesSimulation, self).add_ado(type=DoubleIntegratorDTAgent, **ado_kwargs)
        self._ado_goals.append(ado_kwargs["goal"])

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
        tau = sim_social_forces_tau
        v_0 = sim_social_forces_v_0
        sigma = sim_social_forces_sigma

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
            logging.debug(f"simulation [ado_{iid}]: position={ado_positions[i].data},velocity={ado_velocities[i].data}")
        if ego_position is not None and ego_velocity is not None:
            graph["ego_position"] = ego_position
            graph["ego_velocity"] = ego_velocity
            if not is_intermediate:
                graph["ego_position"].requires_grad = True
            logging.debug(f"simulation [ego]: position={ego_position.data},velocity={ego_velocity.data}")

        # Make graph with resulting force as an output.
        for i in range(num_ados):
            iid = ado_ids[i]

            # Destination force - Force pulling the ado to its assigned goal position.
            direction = torch.sub(graph[f"{iid}_goal"], graph[f"{iid}_position"])
            goal_distance = torch.norm(direction)
            if goal_distance.data < sim_social_forces_min_goal_distance:
                destination_force = torch.zeros(2)
            else:
                direction = torch.div(direction, goal_distance)
                speed = torch.norm(graph[f"{iid}_velocity"])
                destination_force = torch.sub(direction * speed, graph[f"{iid}_velocity"]) * 1 / tau
            logging.debug(f"simulation [ado_{iid}]: destination force -> {destination_force.data}")
            graph[f"{iid}_force"] = destination_force

            # Interactive force - Repulsive potential field by every other agent.
            for j in range(num_ados):
                jid = ado_ids[j]
                if iid == jid:
                    continue
                ij_distance = torch.sub(graph[f"{iid}_position"], graph[f"{jid}_position"]).data
                if np.linalg.norm(ij_distance) > sim_social_forces_max_interaction_distance:
                    v_grad = torch.zeros(2)
                else:
                    v_grad = _repulsive_force(
                        graph[f"{iid}_position"],
                        graph[f"{jid}_position"],
                        graph[f"{iid}_velocity"],
                        graph[f"{jid}_velocity"],
                    )
                graph[f"{iid}_force"] = torch.sub(graph[f"{iid}_force"], v_grad)
                logging.debug(f"simulation [ado_{iid}]: interaction force ado {jid} -> {v_grad.data}")

            # Interactive force w.r.t. ego - Repulsive potential field.
            if ego_position is not None and ego_velocity is not None:
                v_grad = _repulsive_force(
                    graph[f"{iid}_position"], graph["ego_position"], graph[f"{iid}_velocity"], graph["ego_velocity"]
                )
                graph[f"{iid}_force"] = torch.sub(graph[f"{iid}_force"], v_grad)
                logging.debug(f"simulation [ado_{iid}]: interaction force ego -> {v_grad.data}")

            # Summarize (standard) graph elements.
            graph[f"{iid}_force_norm"] = torch.norm(graph[f"{iid}_force"])

        # Graph health check.
        assert self.graph_check(graph=graph)
        return graph

    def build_graph_from_agents(
        self, ados: List[Agent] = None, ego_state: np.ndarray = None
    ) -> Dict[str, torch.Tensor]:
        """Build graph from agents (list of ados and ego state). If the list of ados is set to None the internal ado
        state is used for building the graph. If the ego_state is None, the interaction between the ego and the
        environment is ignored."""
        ados = self._ados if ados is None else ados
        if ego_state is not None:
            assert ego_state.size >= 4, "ego state must contain (x, y) - position and (vx, vy) - velocity"

        ado_positions = [torch.tensor(ado.position.astype(float)) for ado in ados]
        ado_velocities = [torch.tensor(ado.velocity.astype(float)) for ado in ados]

        ego_position = torch.tensor(ego_state[0:2].astype(float)) if ego_state is not None else None
        ego_velocity = torch.tensor(ego_state[3:5].astype(float)) if ego_state is not None else None

        return self.build_graph(ado_positions, ado_velocities, ego_position, ego_velocity)

    def graph_check(self, graph: Dict[str, torch.Tensor]) -> bool:
        """Check healthiness of graph by looking for specific keys in the graph that are required."""
        is_okay = True
        is_okay = is_okay and all([req_key in graph.keys() for req_key in []])
        is_okay = is_okay and all([f"{ado.id}_force_norm" for ado in self._ados])
        return is_okay
