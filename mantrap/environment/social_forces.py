from typing import Dict, List, Tuple, Union

import numpy as np
from scipy.stats import rv_continuous
import torch

from mantrap.agents.agent import Agent
from mantrap.agents import DoubleIntegratorDTAgent
from mantrap.constants import *
from mantrap.environment.environment import GraphBasedEnvironment
from mantrap.environment.iterative import IterativeEnvironment


class SocialForcesEnvironment(IterativeEnvironment):
    """Social Forces Simulation.
    Pedestrian Dynamics based on to "Social Force Model for Pedestrian Dynamics" (D. Helbling, P. Molnar). The idea
    of Social Forces is to determine interaction forces by taking into account the following entities:

    *Goal force*:
    Each ado has a specific goal state/position in mind, to which it moves to. The goal pulling force
    is modelled as correction term between the direction vector of the current velocity and the goal direction
    (vector between current position and goal).

    .. math:: F_{goal} = 1 / tau_{a} (v_a^0 e_a - v_a)

    *Interaction force*:
    For modelling interaction between multiple agents such as avoiding collisions each agent
    has an with increasing distance exponentially decaying repulsion field. Together with the scalar product of the
    velocity of each agent pair (in order to not alter close but non interfering agents, e.g. moving parallel to
    each other) the interaction term is constructed.

    .. math:: V_{aB} (x) = V0_a exp(−x / \sigma_a)
    .. math:: F_{interaction} = - grad_{r_{ab}} V_{aB}(||r_{ab}||)

    To create multi-modality and stochastic effects several sets of environment parameters can be assigned to the ado,
    each representing one of it's modes.
    """

    ###########################################################################
    # Scene ###################################################################
    ###########################################################################
    def add_ado(
        self,
        num_modes: int = 1,
        goal: torch.Tensor = torch.zeros(2),
        v0s: Union[List[Tuple[rv_continuous, Dict[str, float]]], np.ndarray] = None,
        sigmas: Union[List[Tuple[rv_continuous, Dict[str, float]]], np.ndarray] = None,
        weights: np.ndarray = None,
        **ado_kwargs,
    ) -> Agent:
        # Social Forces requires to introduce a goal point, the agent is heading to. Find it in the parameters
        # and add it to the ado parameters dictionary.
        assert goal.size() == torch.Size([2])
        goal = goal.detach().float()

        # In order to introduce multi-modality and stochastic effects the underlying parameters of the social forces
        # environment are sampled from distributions, each for one mode. If not stated the default parameters are
        # used as Gaussian distribution around the default value.
        assert (weights is not None) == (type(v0s) == np.ndarray) == (type(sigmas) == np.ndarray)
        if type(v0s) != np.ndarray:
            v0s, weights_v = self.ado_mode_params(v0s, SOCIAL_FORCES_DEFAULTS[PK_V0], num_modes=num_modes)
            sigmas, weights_s = self.ado_mode_params(sigmas, SOCIAL_FORCES_DEFAULTS[PK_SIGMA], num_modes=num_modes)
            weights = np.multiply(weights_s, weights_v)

        # Fill ghost argument list with mode parameters.
        tau = SOCIAL_FORCES_DEFAULTS[PK_TAU]
        args_list = [{PK_V0: v0s[i], PK_SIGMA: sigmas[i], PK_TAU: tau, PK_GOAL: goal} for i in range(num_modes)]

        # Finally add ado ghosts to environment.
        return super(SocialForcesEnvironment, self).add_ado(
            ado_type=DoubleIntegratorDTAgent,
            num_modes=num_modes,
            weights=weights,
            arg_list=args_list,
            **ado_kwargs
        )

    ###########################################################################
    # Simulation Graph ########################################################
    ###########################################################################
    def build_graph(self, ego_state: torch.Tensor = None, k: int = 0, **graph_kwargs) -> Dict[str, torch.Tensor]:

        # Repulsive force introduced by every other agent (depending on relative position and (!) velocity).
        def _repulsive_force(
            alpha_position: torch.Tensor,
            beta_position: torch.Tensor,
            alpha_velocity: torch.Tensor,
            beta_velocity: torch.Tensor,
            p_v_0: float,
            p_sigma: float,
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
            v = p_v_0 * torch.exp(-b / p_sigma)

            # The repulsive force between agents is the negative gradient of the other (beta -> alpha)
            # potential field. Therefore subtract the gradient of V w.r.t. the relative distance.
            force = torch.autograd.grad(v, relative_distance, create_graph=True)[0]
            if torch.any(torch.isnan(force)):  # TODO: why is nan rarely occurring here and how to deal with that ?
                return torch.zeros(2)
            else:
                return force

        # Graph initialization - Add ados and ego to graph (position, velocity and goals).
        graph_k = self.write_state_to_graph(ego_state, ado_grad=True, k=k, **graph_kwargs)

        # Make graph with resulting force as an output.
        for ghost in self._ado_ghosts:
            tau, v0, sigma = ghost.params[PK_TAU], ghost.params[PK_TAU], ghost.params[PK_SIGMA]
            gpos = graph_k[f"{ghost.id}_{k}_{GK_POSITION}"]
            gvel = graph_k[f"{ghost.id}_{k}_{GK_VELOCITY}"]

            # Destination force - Force pulling the ado to its assigned goal position.
            direction = torch.sub(ghost.params[PK_GOAL], graph_k[f"{ghost.id}_{k}_{GK_POSITION}"])
            goal_distance = torch.norm(direction)
            if goal_distance.item() < SOCIAL_FORCES_MAX_GOAL_DISTANCE:
                destination_force = torch.zeros(2)
            else:
                direction = torch.div(direction, goal_distance)
                speed = torch.norm(graph_k[f"{ghost.id}_{k}_{GK_VELOCITY}"])
                destination_force = torch.sub(direction * speed, graph_k[f"{ghost.id}_{k}_{GK_VELOCITY}"]) * 1 / tau
            graph_k[f"{ghost.id}_{k}_{GK_CONTROL}"] = destination_force

            # Interactive force - Repulsive potential field by every other agent.
            for other in self.ghosts:
                if ghost.agent.id == other.agent.id:  # ghosts from the same parent agent dont repulse each other
                    continue
                distance = torch.sub(graph_k[f"{ghost.id}_{k}_{GK_POSITION}"], graph_k[f"{other.id}_{k}_{GK_POSITION}"])
                if torch.norm(distance) > SOCIAL_FORCES_MAX_INTERACTION_DISTANCE:
                    continue
                else:
                    opos = graph_k[f"{other.id}_{k}_{GK_POSITION}"]
                    ovel = graph_k[f"{other.id}_{k}_{GK_VELOCITY}"]
                    v_grad = _repulsive_force(gpos, opos, gvel, ovel, p_v_0=v0, p_sigma=sigma)
                v_grad = v_grad * other.weight  # weight force by probability of the modes
                graph_k[f"{ghost.id}_{k}_{GK_CONTROL}"] = torch.sub(graph_k[f"{ghost.id}_{k}_{GK_CONTROL}"], v_grad)

            # Interactive force w.r.t. ego - Repulsive potential field.
            if ego_state is not None:
                ego_pos = graph_k[f"ego_{k}_{GK_POSITION}"]
                ego_vel = graph_k[f"ego_{k}_{GK_VELOCITY}"]
                v_grad = _repulsive_force(gpos, ego_pos, gvel, ego_vel, p_v_0=v0, p_sigma=sigma)
                graph_k[f"{ghost.id}_{k}_{GK_CONTROL}"] = torch.sub(graph_k[f"{ghost.id}_{k}_{GK_CONTROL}"], v_grad)

        return graph_k

    ###########################################################################
    # Operators ###############################################################
    ###########################################################################
    def _copy_ados(self, env_copy: GraphBasedEnvironment) -> GraphBasedEnvironment:
        for i in range(self.num_ados):
            ghosts_ado = self.ghosts_by_ado_index(ado_index=i)
            ado_id, _ = self.split_ghost_id(ghost_id=ghosts_ado[0].id)
            env_copy.add_ado(
                position=ghosts_ado[0].agent.position,  # same over all ghosts of same ado
                velocity=ghosts_ado[0].agent.velocity,  # same over all ghosts of same ado
                history=ghosts_ado[0].agent.history,  # same over all ghosts of same ado
                time=self.time,
                weights=np.array([ghost.weight for ghost in ghosts_ado]) if env_copy.is_multi_modal else np.ones(1),
                num_modes=self.num_modes if env_copy.is_multi_modal else 1,
                identifier=self.split_ghost_id(ghost_id=ghosts_ado[0].id)[0],
                goal=ghosts_ado[0].params[PK_GOAL],
                v0s=np.array([ghost.params[PK_V0] for ghost in ghosts_ado]),
                sigmas=np.array([ghost.params[PK_SIGMA] for ghost in ghosts_ado]),
            )
        return env_copy

    ###########################################################################
    # Simulation properties ###################################################
    ###########################################################################
    @property
    def environment_name(self) -> str:
        return "social_forces"

    @property
    def is_multi_modal(self) -> bool:
        return True

    @property
    def is_deterministic(self) -> bool:
        return True

    @property
    def is_differentiable_wrt_ego(self) -> bool:
        return True
