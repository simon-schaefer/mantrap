import typing

import numpy as np
import scipy.stats
import torch

import mantrap.agents
import mantrap.constants

from ..base.graph_based import GraphBasedEnvironment
from ..base.iterative import IterativeEnvironment


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
        v0s: typing.Union[typing.List[typing.Tuple[scipy.stats.rv_continuous, typing.Dict[str, float]]],
                          np.ndarray] = None,
        weights: np.ndarray = None,
        **ado_kwargs
    ) -> mantrap.agents.base.DTAgent:
        # In order to introduce multi-modality and stochastic effects the underlying v0 parameters of the potential
        # field environment are sampled from distributions, each for one mode. If not stated the default parameters
        # are used as Gaussian distribution around the default value.
        assert (weights is not None) == (type(v0s) == np.ndarray)
        if type(v0s) != np.ndarray:
            v0s, weights = self.ado_mode_params(xs=v0s,
                                                x0_default=mantrap.constants.SOCIAL_FORCES_DEFAULT_V0,
                                                num_modes=num_modes)

        # Fill ghost argument list with mode parameters.
        args_list = [{mantrap.constants.PK_V0: v0s[i]} for i in range(num_modes)]

        # Finally add ado ghosts to environment.
        return super(PotentialFieldEnvironment, self).add_ado(
            ado_type=mantrap.agents.DoubleIntegratorDTAgent,
            num_modes=num_modes,
            weights=weights,
            arg_list=args_list,
            **ado_kwargs
        )

    ###########################################################################
    # Simulation Graph ########################################################
    ###########################################################################
    def build_graph(self, ego_state: torch.Tensor = None, k: int = 0,  **graph_kwargs
                    ) -> typing.Dict[str, torch.Tensor]:
        # Graph initialization - Add ados and ego to graph (position, velocity and goals).
        graph = self.write_state_to_graph(ego_state, k=k, **graph_kwargs)

        # Make graph with resulting force as an output.
        for ghost in self.ghosts:
            gpos = graph[f"{ghost.id}_{k}_{mantrap.constants.GK_POSITION}"]
            force = torch.zeros(2)

            if ego_state is not None:
                ego_pos = graph[f"{mantrap.constants.ID_EGO}_{k}_{mantrap.constants.GK_POSITION}"]
                delta = gpos - ego_pos
                delta = torch.sign(delta) * torch.exp(- torch.abs(delta))
                ego_force = ghost.params[mantrap.constants.PK_V0] * delta * ghost.weight
                force = torch.add(force, ego_force)

            graph[f"{ghost.id}_{k}_{mantrap.constants.GK_CONTROL}"] = force

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
                weights=np.array([ghost.weight for ghost in ghosts_ado] if env_copy.is_multi_modal else [1]),
                num_modes=self.num_modes if env_copy.is_multi_modal else 1,
                identifier=self.split_ghost_id(ghost_id=ghosts_ado[0].id)[0],
                v0s=np.array([ghost.params[mantrap.constants.PK_V0] for ghost in ghosts_ado]),
            )
        return env_copy

    ###########################################################################
    # Simulation parameters ###################################################
    ###########################################################################
    @staticmethod
    def environment_name() -> str:
        return "potential_field"

    @property
    def is_multi_modal(self) -> bool:
        return True

    @property
    def is_deterministic(self) -> bool:
        return True

    @property
    def is_differentiable_wrt_ego(self) -> bool:
        return True
