from typing import Dict

import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.agents.agent import Agent
from mantrap.constants import *
from mantrap.environment.iterative import IterativeEnvironment


class KalmanEnvironment(IterativeEnvironment):
    """Kalman (Filter) - based Environment.

    The Kalman environment implements the update rules, defined in the Kalman Filter, to update the agents
    states iteratively. Thereby no interactions between the agents are taken into account.

    Since currently no uncertainty estimates are supported yet, the Kalman update rules break down to a
    simple mean update, based on the agent's dynamics.
    """

    ###########################################################################
    # Scene ###################################################################
    ###########################################################################
    def add_ado(self, **ado_kwargs) -> Agent:
        return super(KalmanEnvironment, self).add_ado(IntegratorDTAgent, **ado_kwargs)

    ###########################################################################
    # Simulation Graph ########################################################
    ###########################################################################
    def build_graph(self, ego_state: torch.Tensor = None, k: int = 0,  **graph_kwargs) -> Dict[str, torch.Tensor]:
        # Graph initialization - Add ados and ego to graph (position, velocity and goals).
        graph = self.write_state_to_graph(ego_state, k=k, **graph_kwargs)

        # Since the ado are assumed to be single integrators, having zero uncertainty in the dynamics and
        # no interaction interferences with other agents the only update the Kalman environment has to do
        # is their state-mean update, with the current position as control input for their dynamics.
        for ghost in self.ghosts:
            graph[f"{ghost.id}_{k}_{GK_CONTROL}"] = graph[f"{ghost.id}_{k}_{GK_VELOCITY}"]

        return graph

    ###########################################################################
    # Simulation parameters ###################################################
    ###########################################################################
    @property
    def environment_name(self) -> str:
        return "kalman"

    @property
    def is_multi_modal(self) -> bool:
        return False

    @property
    def is_deterministic(self) -> bool:
        return True

    @property
    def is_differentiable_wrt_ego(self) -> bool:
        return False
