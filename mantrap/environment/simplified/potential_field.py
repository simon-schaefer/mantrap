from typing import Dict

import torch

from mantrap.agents.agent import Agent
from mantrap.constants import *
from mantrap.environment import SocialForcesEnvironment


class PotentialFieldEnvironment(SocialForcesEnvironment):
    """Simplified version of social forces environment class.

    The simplified model assumes static agents (ados) in the scene, having zero velocity (if not stated otherwise)
    and no incentive to move since goal and position are the same. Hereby, the graph model is cut to the pure
    interaction between ego and ado, no inter-ado interaction and goal pulling force. Since the ados would not move
    at all without an ego agent in the scene, the interaction loss simply comes down the the distance of every position
    of the ados in time to their initial (static) position.
    """

    ###########################################################################
    # Scene ###################################################################
    ###########################################################################
    def add_ado(
        self,
        goal: torch.Tensor = None,
        position: torch.Tensor = torch.zeros(2),
        velocity: torch.Tensor = torch.zeros(2),
        **kwargs
    ) -> Agent:
        """Add another ado agent to the scene. Thereby set the goal to the initial position of the ado, in order to
        enforce a static agent behaviour, by not giving an incentive to move. """
        return super(PotentialFieldEnvironment, self).add_ado(position=position, goal=position, velocity=velocity, **kwargs)

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
                force = torch.sign(delta) * torch.exp(- torch.abs(delta))
                graph[f"{ghost.id}_{k}_{GK_CONTROL}"] = torch.add(graph[f"{ghost.id}_{k}_{GK_CONTROL}"], force)

            # Summarize (standard) graph elements.
            graph[f"{ghost.id}_{k}_{GK_OUTPUT}"] = torch.norm(graph[f"{ghost.id}_{k}_{GK_CONTROL}"])

        return graph

    ###########################################################################
    # Simulation parameters ###################################################
    ###########################################################################
    @property
    def environment_name(self) -> str:
        return "potential_field"

    @property
    def is_multi_modality(self) -> bool:
        return False

    @property
    def is_deterministic(self) -> bool:
        return True
