from collections import namedtuple
import copy
from typing import Any, Dict, List, Tuple, Union

import torch

from mantrap.agents.agent import Agent
from mantrap.agents import DoubleIntegratorDTAgent
from mantrap.constants import (
    sim_x_axis_default,
    sim_y_axis_default,
    sim_dt_default,
    sim_social_forces_defaults,
    sim_social_forces_min_goal_distance,
    sim_social_forces_max_interaction_distance,
)
from mantrap.simulation.graph_based import GraphBasedSimulation
from mantrap.utility.maths import Distribution, Gaussian
from mantrap.utility.shaping import check_ego_controls, check_ego_trajectory


class SocialForcesSimulation(GraphBasedSimulation):

    # Re-Definition of the Ghost object introducing further social-forces specific parameters such as a goal or
    # agent-dependent simulation parameters v0, sigma and tau.
    Ghost = namedtuple("Ghost", "agent goal v0 sigma tau weight id")

    def __init__(
        self,
        ego_type: Agent.__class__ = None,
        ego_kwargs: Dict[str, Any] = None,
        x_axis: Tuple[float, float] = sim_x_axis_default,
        y_axis: Tuple[float, float] = sim_y_axis_default,
        dt: float = sim_dt_default,
    ):
        super(SocialForcesSimulation, self).__init__(ego_type, ego_kwargs, x_axis=x_axis, y_axis=y_axis, dt=dt)
        self._ado_ghosts = []

    def predict_w_controls(self, controls: torch.Tensor, return_more: bool = False, **graph_kwargs) -> torch.Tensor:
        graphs = self.build_connected_graph(ego_controls=controls, ego_grad=False, **graph_kwargs)
        return self.transcribe_graph(graphs, t_horizon=controls.shape[0], returns=return_more)

    def predict_w_trajectory(self, trajectory: torch.Tensor, return_more: bool = False, **graph_kwargs) -> torch.Tensor:
        assert check_ego_trajectory(ego_trajectory=trajectory, pos_and_vel_only=True)
        graphs = self.build_connected_graph(ego_trajectory=trajectory, ego_grad=False, **graph_kwargs)
        return self.transcribe_graph(graphs, t_horizon=trajectory.shape[0], returns=return_more)

    def predict_wo_ego(self, t_horizon: int, return_more: bool = False, **graph_kwargs) -> torch.Tensor:
        graphs = self.build_connected_graph(t_horizon=t_horizon, ego_grad=False, **graph_kwargs)
        return self.transcribe_graph(graphs, t_horizon=t_horizon, returns=return_more)

    def step(self, ego_control: torch.Tensor = None) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        ado_states, ego_next_state = super(SocialForcesSimulation, self).step(ego_control=ego_control)
        for i in range(self.num_ado_ghosts):
            i_ado, i_mode = self.ghost_to_ado_index(i)
            self._ado_ghosts[i].agent.reset(ado_states[i_ado, i_mode, 0, :], history=None)  # new state is appended
        return ado_states, ego_next_state

    def add_ado(
        self,
        goal: torch.Tensor,
        num_modes: int,
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

        # In order to introduce multi-modality and stochastic effects the underlying parameters of the social forces
        # simulation are sampled from distributions, each for one mode. If not stated the default parameters are
        # used as Gaussian distribution around the default value.
        v0s_default = Gaussian(sim_social_forces_defaults["v0"], sim_social_forces_defaults["v0"] / 2)
        v0s = v0s if v0s is not None else [v0s_default] * num_modes
        sigma_default = Gaussian(sim_social_forces_defaults["sigma"], sim_social_forces_defaults["sigma"] / 2)
        sigmas = sigmas if sigmas is not None else [sigma_default] * num_modes
        weights = weights if weights is not None else (torch.ones(num_modes) / num_modes).tolist()
        assert len(v0s) == len(sigmas) == len(weights), "simulation parameters and number of modes do not match"

        # Create ado and ado ghosts.
        super(SocialForcesSimulation, self).add_ado(type=DoubleIntegratorDTAgent, **ado_kwargs)
        for i in range(num_modes):
            assert v0s[i].__class__.__bases__[0] == Distribution
            assert sigmas[i].__class__.__bases__[0] == Distribution
            ado = copy.deepcopy(self._ados[-1])
            v0 = abs(float(v0s[i].sample()))
            sigma = abs(float(sigmas[i].sample()))
            tau = sim_social_forces_defaults["tau"]
            gid = ado.id + "_" + str(i)
            self._ado_ghosts.append(self.Ghost(ado, goal=goal, v0=v0, sigma=sigma, tau=tau, weight=weights[i], id=gid))

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
        graph = super(SocialForcesSimulation, self).build_graph(ego_state, ado_grad=True, **graph_kwargs)
        k = graph_kwargs["k"] if "k" in graph_kwargs.keys() else 0
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
                distance = torch.sub(graph[f"{ghost.id}_{k}_position"], graph[f"{other.id}_{k}_position"]).data
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
        ado_ghosts_copy = copy.deepcopy(self._ado_ghosts)

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
        for ig, ghost in enumerate(self._ado_ghosts):
            self._ado_ghosts[ig].agent.update(graphs[f"{ghost.id}_{t_horizon - 1}_force"], dt=self.dt)
            graphs[f"{ghost.id}_{t_horizon}_position"] = self._ado_ghosts[ig].agent.position
            graphs[f"{ghost.id}_{t_horizon}_velocity"] = self._ado_ghosts[ig].agent.velocity

        # Reset ado ghosts to previous states.
        self._ado_ghosts = ado_ghosts_copy
        return graphs

    ###########################################################################
    # Ado properties ##########################################################
    ###########################################################################

    @property
    def ado_ghosts_agents(self) -> List[Agent]:
        return [ghost.agent for ghost in self._ado_ghosts]

    @property
    def ado_ghosts(self) -> List[Ghost]:
        return self._ado_ghosts

    @property
    def num_ado_ghosts(self) -> int:
        return len(self._ado_ghosts)

    @property
    def num_ado_modes(self) -> int:
        """The number of modes results from the ratio between the number of ado ghosts (pseudo ados, i.e. different
        versions of an ado differentiating by a their parameters) and the number of ados. Thereby we assume that
        all ados have the same number of modes (which is usually the case by construction). """
        if self.num_ados == 0:
            num_ado_modes = 0
        else:
            assert len(self._ado_ghosts) % self.num_ados == 0
            num_ado_modes = int(len(self._ado_ghosts) / self.num_ados)
        return num_ado_modes
