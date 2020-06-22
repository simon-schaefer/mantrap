import typing

import numpy as np
import torch
import torch.distributions

import mantrap.environment

from .acc_interact import InteractionAccelerationModule


class InteractionPositionModule(InteractionAccelerationModule):
    """Loss based on difference of positions due to interaction between robot and ados.

    As a proxy for interaction based on the mean position of every ado is computed in a (fictional) scene without an
    ego (robot) and compared to the actual occurring positions in the scene, as in intuitive measure for the change
    the robot's presence introduces to the scene.

    Re-Predicting it every time-step would be more correct, however it would also require a lot more computational
    effort (horizon times as much to be exact). Therefore merely the behavior of the ado without ego is computed
    that would occur, if the ego is not there from the beginning.

    .. math:: objective = 1/T \\sum_{T} \\sum_{ados} || pos_{t,i} - pos_{t,i}^{wo} ||_2

    :param env: solver's environment environment for predicting the behaviour without interaction.
    """
    def __init__(self, env: mantrap.environment.base.GraphBasedEnvironment, t_horizon: int, weight: float = 1.0,
                 **unused):
        super(InteractionPositionModule, self).__init__(env=env, t_horizon=t_horizon, weight=weight)
        self._max_value = mantrap.constants.OBJECTIVE_POS_INTERACT_MAX

    def summarize_distribution(self, dist_dict: typing.Dict[str, torch.distributions.Distribution]
                               ) -> torch.Tensor:
        """Compute ado-wise positions from velocity distribution dict mean values."""
        _ , ado_states = self.env.states()
        velocities = torch.zeros((self.env.num_ados, self.t_horizon, self.env.num_modes, 2))
        for ado_id, distribution in dist_dict.items():
            m_ado = self.env.index_ado_id(ado_id)
            velocities[m_ado, :, :, :] = distribution.mean
        positions = mantrap.utility.maths.integrate_numerical(velocities, dt=self.env.dt, x0=ado_states[:, 0:2])
        return positions

    ###########################################################################
    # Objective Properties ####################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "interaction_pos"
