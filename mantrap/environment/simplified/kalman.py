import typing

import torch

import mantrap.agents
import mantrap.constants
import mantrap.environment.intermediates


class KalmanEnvironment(mantrap.environment.intermediates.IterativeEnvironment):
    """Kalman (Filter) - based Environment.

    The Kalman environment implements the update rules, defined in the Kalman Filter, to update the agents
    states iteratively. Thereby no interactions between the agents are taken into account.

    Since currently no uncertainty estimates are supported yet, the Kalman update rules break down to a
    simple mean update, based on the agent's dynamics.
    """

    ###########################################################################
    # Scene ###################################################################
    ###########################################################################
    def add_ado(self, **ado_kwargs) -> mantrap.agents.DTAgent:
        return super(KalmanEnvironment, self).add_ado(mantrap.agents.IntegratorDTAgent, **ado_kwargs)

    ###########################################################################
    # Simulation Graph ########################################################
    ###########################################################################
    def build_graph(self, ego_state: torch.Tensor = None, k: int = 0,  **graph_kwargs
                    ) -> typing.Dict[str, torch.Tensor]:
        # Graph initialization - Add ados and ego to graph (position, velocity and goals).
        graph = self.write_state_to_graph(ego_state, k=k, **graph_kwargs)

        # Since the ado are assumed to be single integrators, having zero uncertainty in the dynamics and
        # no interaction interferences with other agents the only update the Kalman environment has to do
        # is their state-mean update, with the current position as control input for their dynamics.
        for ghost in self.ghosts:
            ghost_velocity_k = graph[f"{ghost.id}_{k}_{mantrap.constants.GK_VELOCITY}"]
            graph[f"{ghost.id}_{k}_{mantrap.constants.GK_CONTROL}"] = ghost_velocity_k

        return graph

    ###########################################################################
    # Simulation parameters ###################################################
    ###########################################################################
    @staticmethod
    def environment_name() -> str:
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
