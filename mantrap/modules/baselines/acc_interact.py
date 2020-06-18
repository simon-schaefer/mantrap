import typing

import numpy as np
import torch
import torch.distributions

import mantrap.constants
import mantrap.environment

from ..base import PureObjectiveModule


class InteractionAccelerationModule(PureObjectiveModule):
    """Loss based on accelerational interaction between robot and ados.

    As a proxy for interaction based on the acceleration of every ado is computed in a (fictional) scene without an
    ego (robot) and compared to the actual occurring accelerations in the scene. As for autonomous driving the
    acceleration can be expressed "moving comfort", since a change in acceleration, especially a sudden change like
    strong de-acceleration, decreases the comfort of the agent.

    Re-Predicting it every time-step would be more correct, however it would also require a lot more computational
    effort (horizon times as much to be exact). Therefore merely the behavior of the ado without ego is computed
    that would occur, if the ego is not there from the beginning.

    .. math:: objective = 1/T \\sum_{T} \\sum_{ados} || acc_{t,i} - acc_{t,i}^{wo} ||_2

    :param env: environment for predicting the behaviour without interaction.
    """
    def __init__(self, env: mantrap.environment.base.GraphBasedEnvironment, t_horizon: int, weight: float = 1.0,
                 **unused):
        super(InteractionAccelerationModule, self).__init__(env=env, t_horizon=t_horizon, weight=weight)
        if env.num_ados > 0:
            self._derivative_2 = mantrap.utility.maths.Derivative2(horizon=self.t_horizon + 1,
                                                                   dt=self.env.dt,
                                                                   num_axes=2)
            dist_dict = self.env.compute_distributions_wo_ego(t_horizon=self.t_horizon)
            self._ado_accelerations_wo = self.distribution_to_acceleration(dist_dict)

    def _objective_core(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str], tag: str
                        ) -> typing.Union[torch.Tensor, None]:
        """Determine objective value core method.

        To compute the objective value first predict the behaviour of all agents in the scene in the planning
        horizon, conditioned on the ego trajectory. Then compare the (mode-wise if multi-modal) means of
        the previously computed un-conditioned (see initialization) and the conditioned distribution, in
        terms of their second derivative, i.e. acceleration.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        :param tag: name of optimization call (name of the core).
        """
        # The objective can only work if any ado agents are taken into account, otherwise return None.
        if len(ado_ids) == 0 or self._env.num_ados == 0:
            return None

        # If more than zero ado agents are taken into account, compute the objective as described.
        # It is important to take all agents into account during the environment forward prediction step
        # (`compute_distributions()`) to not introduce possible behavioural changes into the forward prediction,
        # which occur due to a reduction of the agents in the scene.
        dist_dict = self.env.compute_distributions(ego_trajectory=ego_trajectory)
        acceleration = self.distribution_to_acceleration(dist_dict)

        # Average of all ados that should be taken into account.
        cost = torch.zeros(1)
        for ado_id in ado_ids:
            m_ado = self.env.index_ado_id(ado_id=ado_id)
            cost_ado = torch.norm(acceleration[m_ado, :, :] - self._ado_accelerations_wo[m_ado, :, :], dim=1)
            cost += torch.mean(cost_ado)
        cost = cost / len(ado_ids)  # average of pedestrians.

        # Clamp maximal value (minimal is zero anyways, due to L2-norm).
        return cost.clamp_max(mantrap.constants.OBJECTIVE_ACC_INTERACT_MAX)

    def distribution_to_acceleration(self, dist_dict: typing.Dict[str, torch.distributions.Distribution]
                                     ) -> torch.Tensor:
        """Compute ado-wise accelerations from positional distribution dict mean values."""
        sample_length = self.env.num_modes * (self.t_horizon + 1)
        accelerations = torch.zeros((self.env.num_ados, sample_length, 2))
        for ado_id, distribution in dist_dict.items():
            m_ado = self.env.index_ado_id(ado_id)
            accelerations[m_ado, :, :] = self._derivative_2.compute(distribution.mean).view(-1, 2)
        return accelerations

    def normalize(self, x: typing.Union[np.ndarray, float]) -> typing.Union[np.ndarray, float]:
        """Normalize the objective/constraint value for improved optimization performance.

        The objective value is clamped to some maximal value which enforces the resulting objective
        values to be in some range. And can hence serve as normalize factor.

        :param x: objective/constraint value in normal value range.
        :returns: normalized objective/constraint value in range [-1, 1].
        """
        return x / mantrap.constants.OBJECTIVE_ACC_INTERACT_MAX

    def gradient_condition(self) -> bool:
        """Condition for back-propagating through the objective/constraint in order to obtain the
        objective's gradient vector/jacobian (numerically). If returns True and the ego_trajectory
        itself requires a gradient, the objective/constraint value, stored from the last computation
        (`_current_`-variables) has to require a gradient as well.

        If the internal environment is itself differentiable with respect to the ego (trajectory) input, the
        resulting objective value must have a gradient as well.
        """
        return self._env.is_differentiable_wrt_ego

    ###########################################################################
    # Objective Properties ####################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "interact_acc"
