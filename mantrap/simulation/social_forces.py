from collections import namedtuple
import copy
from typing import Any, Dict, List, Tuple, Union

import torch

from mantrap.agents.agent import Agent
from mantrap.agents import IntegratorDTAgent, DoubleIntegratorDTAgent
from mantrap.constants import (
    sim_x_axis_default,
    sim_y_axis_default,
    sim_dt_default,
    sim_social_forces_default_params,
    sim_social_forces_min_goal_distance,
    sim_social_forces_max_interaction_distance,
)
from mantrap.utility.shaping import check_ego_trajectory, check_trajectories, check_policies, check_weights
from mantrap.utility.maths import Distribution, DirecDelta
from mantrap.utility.utility import build_trajectory_from_path
from mantrap.simulation.graph_based import GraphBasedSimulation


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

    def predict(
        self, graph_input: Union[int, torch.Tensor], returns: bool = False, **graph_kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """For predicting the ado states in future time-steps up to t* = t_horizon given some ego trajectory
        subsequently build a graph representation of the current scene in order to predict all states in the next scene,
        using the determined forces as control input for the ado state updates. Then, after iterating through all
        time-steps up to t* = t_horizon, build the ado trajectories by using their state histories. In order to account
        for different ado modes, simulate them as "ghost" agents independently. If no ego_trajectory is given, the
        ego will be ignored instead, and the scene is being forward simulated `t_horizon = graph_input` steps.

        :param graph_input: either ego trajectory or t_horizon.
        :param returns: return the actual system inputs (at every time -> trajectory) and probabilities of each mode.
        :return: predicted trajectories for ados in the scene (either one or multiple for each ado).
        """
        ignore_ego = type(graph_input) != torch.Tensor
        if not ignore_ego:
            if graph_input.shape[1] == 2:
                graph_input = build_trajectory_from_path(positions=graph_input, dt=self.dt, t_start=self.sim_time)
            assert check_ego_trajectory(ego_trajectory=graph_input, pos_and_vel_only=True)
        t_horizon = graph_input if ignore_ego else graph_input.shape[0]

        # Build up simulation graph.
        graphs = self.build_connected_graph(graph_input=graph_input, ego_grad=False, **graph_kwargs)

        # Remodel simulation outputs, as they are all stored in the simulation graph.
        forces = torch.zeros((self.num_ados, self.num_ado_modes, t_horizon, 2))
        trajectories = torch.zeros((self.num_ados, self.num_ado_modes, t_horizon, 5))
        weights = torch.zeros((self.num_ados, self.num_ado_modes))
        for i, ghost in enumerate(self._ado_ghosts):
            i_ado, i_mode = self.ghost_to_ado_index(i)
            ghost_id = self.ado_ghosts[i].id
            for k in range(t_horizon):
                trajectories[i_ado, i_mode, k, 0:2] = graphs[f"{ghost_id}_{k}_position"]
                trajectories[i_ado, i_mode, k, 2:4] = graphs[f"{ghost_id}_{k}_velocity"]
                trajectories[i_ado, i_mode, k, -1] = self.sim_time + self.dt * k
                forces[i_ado, i_mode, k, :] = graphs[f"{ghost_id}_{k}_force"]
            weights[i_ado, i_mode] = ghost.weight

        assert check_policies(forces, num_ados=self.num_ados, num_modes=self.num_ado_modes, t_horizon=t_horizon)
        assert check_weights(weights, num_ados=self.num_ados, num_modes=self.num_ado_modes)
        assert check_trajectories(trajectories, self.num_ados, t_horizon=t_horizon, modes=self.num_ado_modes)
        return trajectories if not returns else (trajectories, forces, weights)

    def step(self, ego_policy: torch.Tensor = None) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        ado_states, ego_next_state = super(SocialForcesSimulation, self).step(ego_policy=ego_policy)
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
        # used as hot-encoded (i.e. direc delta) distribution.
        v0s = v0s if v0s is not None else [DirecDelta(sim_social_forces_default_params["v0"])] * num_modes
        sigmas = sigmas if sigmas is not None else [DirecDelta(sim_social_forces_default_params["sigma"])] * num_modes
        weights = weights if weights is not None else (torch.ones(num_modes) / num_modes).tolist()
        assert len(v0s) == len(sigmas) == len(weights), "simulation parameters and number of modes do not match"

        # Create ado and ado ghosts.
        super(SocialForcesSimulation, self).add_ado(type=DoubleIntegratorDTAgent, **ado_kwargs)
        for i in range(num_modes):
            assert v0s[i].__class__.__bases__[0] == Distribution
            assert sigmas[i].__class__.__bases__[0] == Distribution
            ado = copy.deepcopy(self._ados[-1])
            v0 = float(v0s[i].sample())
            sigma = float(sigmas[i].sample())
            tau = sim_social_forces_default_params["tau"]
            id = ado.id + "_" + str(i)
            self._ado_ghosts.append(self.Ghost(ado, goal=goal, v0=v0, sigma=sigma, tau=tau, weight=weights[i], id=id))

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
                v_grad = v_grad * other.weight  # weight force by probability of the mode
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

    def build_connected_graph(self, graph_input: Union[int, torch.Tensor], **graph_kwargs) -> Dict[str, torch.Tensor]:
        super(SocialForcesSimulation, self).build_connected_graph(graph_input, **graph_kwargs)
        ado_ghosts_copy = copy.deepcopy(self._ado_ghosts)

        # Depending on whether the prediction time horizon or the ego trajectory is given, the ego agent is either
        # ignored or taken into account for simulation.
        ignore_ego = type(graph_input) != torch.Tensor
        t_horizon = graph_input if ignore_ego else graph_input.shape[0]
        ego_first_state = None if ignore_ego else self.ego.state
        ego_states = None
        if not ignore_ego:
            assert self.ego.__class__ == IntegratorDTAgent, "currently only single integrator egos are supported"
            ego_states = build_trajectory_from_path(graph_input[:, 0:2], dt=self.dt, t_start=self.sim_time)

        # Build first graph for the next state, in case no forces are applied on the ego.
        graphs = self.build_graph(ego_state=ego_first_state, k=0, **graph_kwargs)

        # Build the graph iteratively for the whole prediction horizon.
        # Social forces assumes all agents to be controlled by some force vector, i.e. to be double integrators.
        assert all([ghost.agent.__class__ == DoubleIntegratorDTAgent for ghost in self._ado_ghosts])
        for k in range(1, t_horizon):
            for ig, ghost in enumerate(self._ado_ghosts):
                self._ado_ghosts[ig].agent.update(graphs[f"{ghost.id}_{k - 1}_force"], dt=self.dt)

            # The ego movement is, of cause, unknown, since we try to find it here. Therefore motion primitives are
            # used for the ego motion, as guesses for the final trajectory i.e. starting points for optimization.
            graph_k = self.build_graph(ego_states[k, :] if not ignore_ego else None, k=k, **graph_kwargs)
            graphs.update(graph_k)

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
