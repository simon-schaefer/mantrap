from collections import namedtuple
from copy import deepcopy
from typing import Dict, List

import torch

from mantrap.agents import DoubleIntegratorDTAgent
from mantrap.constants import (
    sim_social_forces_defaults,
    sim_social_forces_min_goal_distance,
    sim_social_forces_max_interaction_distance,
)
from mantrap.simulation.simulation import GraphBasedSimulation
from mantrap.utility.maths import Distribution, Gaussian
from mantrap.utility.io import dict_value_or_default
from mantrap.utility.shaping import check_ego_trajectory


class SocialForcesSimulation(GraphBasedSimulation):
    """
    Social Forces Simulation.
    Pedestrian Dynamics based on to "Social Force Model for Pedestrian Dynamics" (D. Helbling, P. Molnar).
    """
    # Re-Definition of the Ghost object introducing further social-forces specific parameters such as a goal or
    # agent-dependent simulation parameters v0, sigma and tau.
    Ghost = namedtuple("Ghost", "agent goal v0 sigma tau weight id")

    ###########################################################################
    # Prediction ##############################################################
    ###########################################################################
    def predict_w_controls(self, controls: torch.Tensor, return_more: bool = False, **graph_kwargs) -> torch.Tensor:
        graphs = self.build_connected_graph(ego_controls=controls, ego_grad=False, **graph_kwargs)
        return self.transcribe_graph(graphs, t_horizon=controls.shape[0] + 1, returns=return_more)

    def predict_w_trajectory(self, trajectory: torch.Tensor, return_more: bool = False, **graph_kwargs) -> torch.Tensor:
        assert check_ego_trajectory(ego_trajectory=trajectory, pos_and_vel_only=True)
        graphs = self.build_connected_graph(ego_trajectory=trajectory, ego_grad=False, **graph_kwargs)
        return self.transcribe_graph(graphs, t_horizon=trajectory.shape[0], returns=return_more)

    def predict_wo_ego(self, t_horizon: int, return_more: bool = False, **graph_kwargs) -> torch.Tensor:
        graphs = self.build_connected_graph(t_horizon=t_horizon, ego_grad=False, **graph_kwargs)
        return self.transcribe_graph(graphs, t_horizon=t_horizon, returns=return_more)

    ###########################################################################
    # Scene ###################################################################
    ###########################################################################
    def add_ado(
        self,
        num_modes: int = 1,
        goal: torch.Tensor = torch.zeros(2),
        v0s: List[Distribution] = None,
        sigmas: List[Distribution] = None,
        weights: List[float] = None,
        **ado_kwargs,
    ):
        """Add ado to the simulation. To create multi-modality and stochastic effects several sets of simulation
        parameters can be assigned to the ado, each representing one of it's modes, and weights by the given
        weight vector, representing the probability of each mode occurring. The ado_kwargs are basis ado parameters
        such as position, velocity, id, etc.
        """
        # Social Forces requires to introduce a goal point, the agent is heading to. Find it in the parameters
        # and add it to the ado parameters dictionary.
        assert goal.size() == torch.Size([2]), "goal position must be two-dimensional (x, y)"
        goal = goal.detach().float()

        # In order to introduce multi-modality and stochastic effects the underlying parameters of the social forces
        # simulation are sampled from distributions, each for one mode. If not stated the default parameters are
        # used as Gaussian distribution around the default value.
        v0s_default = Gaussian(sim_social_forces_defaults["v0"], sim_social_forces_defaults["v0"] / 2)
        v0s = v0s if v0s is not None else [v0s_default] * num_modes
        sigma_default = Gaussian(sim_social_forces_defaults["sigma"], sim_social_forces_defaults["sigma"] / 2)
        sigmas = sigmas if sigmas is not None else [sigma_default] * num_modes
        assert len(v0s) == len(sigmas)

        # For each mode sample new parameters from the previously defined distribution.
        args_list = []
        for i in range(num_modes):
            assert v0s[i].__class__.__bases__[0] == Distribution
            assert sigmas[i].__class__.__bases__[0] == Distribution
            v0 = abs(float(v0s[i].sample()))
            sigma = abs(float(sigmas[i].sample()))
            tau = sim_social_forces_defaults["tau"]
            args_list.append({"v0": v0, "sigma": sigma, "tau": tau, "goal": goal})

        # Finally add ado ghosts to simulation.
        super(SocialForcesSimulation, self).add_ado(
            type=DoubleIntegratorDTAgent,
            num_modes=num_modes,
            weights=weights,
            arg_list=args_list,
            **ado_kwargs
        )

    ###########################################################################
    # Simulation Graph ########################################################
    ###########################################################################
    def build_graph(self, ego_state: torch.Tensor = None, **graph_kwargs) -> Dict[str, torch.Tensor]:

        # Repulsive force introduced by every other agent (depending on relative position and (!) velocity).
        def _repulsive_force(
            alpha_position: torch.Tensor,
            beta_position: torch.Tensor,
            alpha_velocity: torch.Tensor,
            beta_velocity: torch.Tensor,
            v_0: float,
            sigma: float,
        ) -> torch.Tensor:

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
        graph = self.write_state_to_graph(ego_state, ado_grad=True, **graph_kwargs)
        k = dict_value_or_default(graph_kwargs, key="k", default=0)
        for ghost in self.ado_ghosts:
            graph[f"{ghost.id}_{k}_goal"] = ghost.goal

        # Make graph with resulting force as an output.
        for ghost in self._ado_ghosts:
            gpos, gvel = graph[f"{ghost.id}_{k}_position"], graph[f"{ghost.id}_{k}_velocity"]

            # Destination force - Force pulling the ado to its assigned goal position.
            direction = torch.sub(graph[f"{ghost.id}_{k}_goal"], graph[f"{ghost.id}_{k}_position"])
            goal_distance = torch.norm(direction)
            if goal_distance.data < sim_social_forces_min_goal_distance:
                destination_force = torch.zeros(2)
            else:
                direction = torch.div(direction, goal_distance)
                speed = torch.norm(graph[f"{ghost.id}_{k}_velocity"])
                destination_force = torch.sub(direction * speed, graph[f"{ghost.id}_{k}_velocity"]) * 1 / ghost.tau
            graph[f"{ghost.id}_{k}_force"] = destination_force

            # Interactive force - Repulsive potential field by every other agent.
            for other in self._ado_ghosts:
                if ghost.agent.id == other.agent.id:  # ghosts from the same parent agent dont repulse each other
                    continue
                distance = torch.sub(graph[f"{ghost.id}_{k}_position"], graph[f"{other.id}_{k}_position"])
                if torch.norm(distance) > sim_social_forces_max_interaction_distance:
                    continue
                else:
                    opos, ovel = graph[f"{other.id}_{k}_position"], graph[f"{other.id}_{k}_velocity"]
                    v_grad = _repulsive_force(gpos, opos, gvel, ovel, v_0=ghost.v0, sigma=ghost.sigma)
                v_grad = v_grad * other.weight  # weight force by probability of the modes
                graph[f"{ghost.id}_{k}_force"] = torch.sub(graph[f"{ghost.id}_{k}_force"], v_grad)

            # Interactive force w.r.t. ego - Repulsive potential field.
            if ego_state is not None:
                ego_pos, ego_vel = graph[f"ego_{k}_position"], graph[f"ego_{k}_velocity"]
                v_grad = _repulsive_force(gpos, ego_pos, gvel, ego_vel, v_0=ghost.v0, sigma=ghost.sigma)
                graph[f"{ghost.id}_{k}_force"] = torch.sub(graph[f"{ghost.id}_{k}_force"], v_grad)

            # Summarize (standard) graph elements.
            graph[f"{ghost.id}_{k}_output"] = torch.norm(graph[f"{ghost.id}_{k}_force"])

        # Check healthiness of graph by looking for specific keys in the graph that are required.
        assert all([f"{ghost.id}_{k}_output" for ghost in self._ado_ghosts])
        return graph

    ###########################################################################
    # Simulation Graph over time-horizon ######################################
    ###########################################################################
    def build_connected_graph(self, **kwargs) -> Dict[str, torch.Tensor]:
        assert any([x in kwargs.keys() for x in ["ego_controls", "ego_trajectory", "t_horizon"]])

        if "ego_controls" in kwargs.keys():
            ego_states = self.ego.unroll_trajectory(controls=kwargs["ego_controls"], dt=self.dt)
            assert check_ego_trajectory(ego_states, pos_and_vel_only=True)
            return self._build_connected_graph(t_horizon=kwargs["ego_controls"].shape[0], ego_states=ego_states)

        elif "ego_trajectory" in kwargs.keys():
            ego_trajectory = kwargs["ego_trajectory"]
            assert check_ego_trajectory(ego_trajectory, pos_and_vel_only=True)
            return self._build_connected_graph(t_horizon=ego_trajectory.shape[0], ego_states=ego_trajectory)

        else:
            return self._build_connected_graph(t_horizon=kwargs["t_horizon"], ego_states=[None] * kwargs["t_horizon"])

    def _build_connected_graph(self, t_horizon: int, ego_states: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        ado_ghosts_copy = deepcopy(self._ado_ghosts)

        # Build first graph for the next state, in case no forces are applied on the ego.
        graphs = self.build_graph(ego_state=ego_states[0], k=0, **kwargs)

        # Build the graph iteratively for the whole prediction horizon.
        # Social forces assumes all agents to be controlled by some force vector, i.e. to be double integrators.
        assert all([ghost.agent.__class__ == DoubleIntegratorDTAgent for ghost in self._ado_ghosts])
        for k in range(1, t_horizon):
            for ig, ghost in enumerate(self._ado_ghosts):
                self._ado_ghosts[ig].agent.update(graphs[f"{ghost.id}_{k - 1}_force"], dt=self.dt)

            # The ego movement is, of cause, unknown, since we try to find it here. Therefore motion primitives are
            # used for the ego motion, as guesses for the final trajectory i.e. starting points for optimization.
            graph_k = self.build_graph(ego_states[k], k=k, **kwargs)
            graphs.update(graph_k)

        # Update graph for a last time using the forces determined in the previous step.
        for ig in range(self.num_ado_ghosts):
            ghost_id = self.ado_ghosts[ig].id
            self._ado_ghosts[ig].agent.update(graphs[f"{ghost_id}_{t_horizon - 1}_force"], dt=self.dt)
            graphs[f"{ghost_id}_{t_horizon}_position"] = self.ado_ghosts[ig].agent.position
            graphs[f"{ghost_id}_{t_horizon}_velocity"] = self.ado_ghosts[ig].agent.velocity

        # Reset ado ghosts to previous states.
        self._ado_ghosts = ado_ghosts_copy
        return graphs

    ###########################################################################
    # Simulation parameters ###################################################
    ###########################################################################
    @property
    def simulation_name(self) -> str:
        return "social_forces"
