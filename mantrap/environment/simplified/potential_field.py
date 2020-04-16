from typing import Dict, List, Union

import torch

from mantrap.agents.agent import Agent
from mantrap.agents import DoubleIntegratorDTAgent
from mantrap.constants import *
from mantrap.environment.environment import GraphBasedEnvironment
from mantrap.environment.iterative import IterativeEnvironment
from mantrap.utility.maths import Distribution, Gaussian


class PotentialFieldEnvironment(IterativeEnvironment):
    """Simplified version of social forces environment class.

    The simplified model assumes static agents (ados) in the scene, having zero velocity (if not stated otherwise)
    and staying in this (non)-movement since no forces are applied. Hereby, the graph model is cut to the pure
    interaction between ego and ado, no inter-ado interaction and goal pulling force. Since the ados would not move
    at all without an ego agent in the scene, the interaction loss simply comes down the the distance of every position
    of the ados in time to their initial (static) position.
    """

    ###########################################################################
    # Scene ###################################################################
    ###########################################################################
    def add_ado(
        self,
        num_modes: int = 1,
        v0s: Union[List[Distribution], List[float]] = None,
        weights: List[float] = None,
        **ado_kwargs
    ) -> Agent:
        # In order to introduce multi-modality and stochastic effects the underlying v0 parameters of the potential
        # field environment are sampled from distributions, each for one mode. If not stated the default parameters
        # are used as Gaussian distribution around the default value.
        v0s_default = Gaussian(ENV_POTENTIAL_FIELD_V0_DEFAULT, ENV_POTENTIAL_FIELD_V0_DEFAULT / 2)
        v0s = v0s if v0s is not None else [v0s_default] * num_modes

        # For each mode sample new parameters from the previously defined distribution. If no distribution is defined,
        # but a list of floating point values is passed directly, just use them.
        args_list = []
        for i in range(num_modes):
            if type(v0s[i]) == float:
                v0 = v0s[i]
            elif v0s[i].__class__.__bases__[0] == Distribution:
                v0 = abs(float(v0s[i].sample()))
            else:
                raise ValueError
            args_list.append({PARAMS_V0: v0})

        # Finally add ado ghosts to environment.
        return super(PotentialFieldEnvironment, self).add_ado(
            ado_type=DoubleIntegratorDTAgent,
            num_modes=num_modes,
            weights=weights,
            arg_list=args_list,
            **ado_kwargs
        )

    ###########################################################################
    # Simulation Graph ########################################################
    ###########################################################################
    def build_graph(self, ego_state: torch.Tensor = None, k: int = 0,  **graph_kwargs) -> Dict[str, torch.Tensor]:
        # Graph initialization - Add ados and ego to graph (position, velocity and goals).
        graph = self.write_state_to_graph(ego_state, k=k, **graph_kwargs)
        for ghost in self.ghosts:
            graph[f"{ghost.id}_{k}_{GK_GOAL}"] = ghost.params[PARAMS_GOAL]

        # Make graph with resulting force as an output.
        for ghost in self.ghosts:
            gpos = graph[f"{ghost.id}_{k}_{GK_POSITION}"]
            graph[f"{ghost.id}_{k}_{GK_CONTROL}"] = torch.zeros(2)

            if ego_state is not None:
                ego_pos = graph[f"{ID_EGO}_{k}_{GK_POSITION}"]
                delta = gpos - ego_pos
                force = ghost.params[PARAMS_V0] * torch.sign(delta) * torch.exp(- torch.abs(delta)) * ghost.weight
                graph[f"{ghost.id}_{k}_{GK_CONTROL}"] = torch.add(graph[f"{ghost.id}_{k}_{GK_CONTROL}"], force)

            # Summarize (standard) graph elements.
            graph[f"{ghost.id}_{k}_{GK_OUTPUT}"] = torch.norm(graph[f"{ghost.id}_{k}_{GK_CONTROL}"])

        return graph

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
                weights=[ghost.weight for ghost in ghosts_ado],
                num_modes=self.num_modes,
                identifier=self.split_ghost_id(ghost_id=ghosts_ado[0].id)[0],
                v0s=[ghost.params[PARAMS_V0] for ghost in ghosts_ado],
            )
        return env_copy

    ###########################################################################
    # Simulation parameters ###################################################
    ###########################################################################
    @property
    def environment_name(self) -> str:
        return "potential_field"

    @property
    def is_multi_modal(self) -> bool:
        return True

    @property
    def is_deterministic(self) -> bool:
        return True
