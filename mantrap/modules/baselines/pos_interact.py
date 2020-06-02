import typing

import torch
import torch.distributions

import mantrap.environment

from ..base import PureObjectiveModule


class InteractionPositionModule(PureObjectiveModule):
    """Loss based on positional interaction between robot and ados.

    As a proxy for interaction based on the position of every ado is computed in a (fictional) scene without an
    ego (robot) and compared to the actual occurring positions in the scene, as in intuitive measure for the change
    the robot's presence introduces to the scene.

    Re-Predicting it every time-step would be more correct, however it would also require a lot more computational
    effort (horizon times as much to be exact). Therefore merely the behavior of the ado without ego is computed
    that would occur, if the ego is not there from the beginning.

    .. math:: objective = \\sum_{T} \\sum_{ados} || pos_{t,i} - pos_{t,i}^{wo} ||_2

    :param env: solver's environment environment for predicting the behaviour without interaction.
    """
    def __init__(self, env: mantrap.environment.base.GraphBasedEnvironment, t_horizon: int, weight: float = 1.0,
                 **unused):
        super(InteractionPositionModule, self).__init__(env=env, t_horizon=t_horizon, weight=weight)

        if env.is_multi_modal:
            raise NotImplementedError

        if env.num_ados > 0:
            self._ado_positions_wo = self._env.predict_wo_ego(t_horizon=self.t_horizon + 1)[:, :, :, 0:2]

    def objective_core(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str], tag: str
                       ) -> typing.Union[torch.Tensor, None]:
        """Determine objective value core method.

        To compute the objective value first predict the behaviour of all agents (and modes) in the scene in the
        planning horizon, conditioned on the ego trajectory. Then iterate over every ghost in the scene and
        find the deviation between the positions of a specific agent at the specific point in time conditioned
        on the ego trajectory and unconditioned. Multiply by the weights of the modes, in order to encounter for
        difference in importance between these modes.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        :param tag: name of optimization call (name of the core).
        """
        # The objective can only work if any ado agents are taken into account, otherwise return None.
        if len(ado_ids) == 0 or self.env.num_ados == 0:
            return None

        # If more than zero ado agents are taken into account, compute the objective as described.
        # It is important to take all agents into account during the environment forward prediction step
        # (`compute_distributions()`) to not introduce possible behavioural changes into the forward prediction,
        # which occur due to a reduction of the agents in the scene.
        dist_dict = self.env.compute_distributions(ego_trajectory=ego_trajectory)
        positions = self.distribution_to_positions(dist_dict)
        return torch.sum(torch.norm(positions - self._ado_positions_wo, dim=-1))

    def distribution_to_positions(self, dist_dict: typing.Dict[str, torch.distributions.Distribution]
                                  ) -> torch.Tensor:
        """Compute ado-wise positions from positional distribution dict mean values."""
        positions = torch.zeros((self.env.num_ados, self.t_horizon, 2))
        for ado_id, distribution in dist_dict.items():
            m_ado = self.env.index_ado_id(ado_id)
            positions[m_ado, :, :] = distribution.mean
        return positions

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
        return "interaction_pos"
