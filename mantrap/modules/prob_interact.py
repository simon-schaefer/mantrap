import typing

import numpy as np
import torch

import mantrap.environment

from .base import PureObjectiveModule


class InteractionProbabilityModule(PureObjectiveModule):
    """Loss based on unconditioned probability value in distribution conditioned on ego motion.

    The general idea of an interactive objective module is to compare the ado trajectory distributions as if
    no ego/robot would be in the scene with the distributions conditioned on the robot trajectory, in order to
    drive the robot's trajectory optimisation to some state in which the ados are (in average) minimally
    disturbed by the robots motion. For general ado trajectory probability distributions including multi-modality,
    this is not a trivial problem. In fact its is not analytically solved for most multi-modal distributions.
    However in the following some common approaches:

    1) KL-Divergence: The KL-Divergence expresses the similarity some distribution q with respect to another
    distribution p:

    .. math::D_{KL} = \\int q(x) log \\frac{q(x)}{p(x)} dx

    While this is a well-defined and  commonly used similarity measure for "simple" distributions for more
    complex ones, such as GMM (e.g. as Trajectron's output) it is not analytically defined. Methods to
    approximate the KL-Divergence for GMMs embrace Monte Carlo sampling, optimisation (itself) and several
    more which however are not computationally feasible for an online application, especially since the
    objective's gradient has to be computed. Other methods simply the real GMM to a single Gaussian, by a
    weighted average over its parameters, which is a) not guaranteed to be a meaningful distribution and
    b) looses the advantages of predicting multi-modal distributions in the first place.

    Especially for trajectron one could also not use the output distribution, but some intermediate (maybe
    simpler distribution) instead, e.g. the latent space, but this one does not depend on the ego trajectory
    so cannot be used for its optimisation.

    2) Unconditioned path projection: Another approach is to compute (and maximize) the probability of the
    mean un-conditioned trajectories (mode-wise) appearing in the conditioned distribution. While it only takes
    into account the mean values (and weights) it is very efficient to compute while still taking the full
    conditioned distribution into account and has shown to be "optimise-able" in the training of Trajectron.
    Since the distributions itself are constant, while the sampled trajectories vary, the objective is also
    constant regarding the same scenario, which also improves its "optimise-ability".

    :param env: solver's environment environment for predicting the behaviour without interaction.
    """

    def __init__(self, env: mantrap.environment.base.GraphBasedEnvironment, t_horizon: int, weight: float = 1.0,
                 **unused):
        super(InteractionProbabilityModule, self).__init__(env=env, t_horizon=t_horizon, weight=weight)

        # Determine mean trajectories and weights of unconditioned distribution. Therefore compute the
        # unconditioned distribution and store the resulting values in an ado-id-keyed dictionary.
        if env.num_ados > 0:
            self._dist_un_conditioned = env.compute_distributions_wo_ego(t_horizon)

    def _objective_core(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str], tag: str
                        ) -> typing.Union[torch.Tensor, None]:
        """Determine objective value core method.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        :param tag: name of optimization call (name of the core).
        """
        # The objective can only work if any ado agents are taken into account, otherwise return None.
        # If the environment is not deterministic the output distribution is not defined, hence return None.
        if len(ado_ids) == 0 or self.env.num_ados == 0:
            return None

        # Compute the conditioned distribution. Then for every ado in the ado_ids`-list determine the `
        # probability of occurring in this distribution, using the distributions core methods.
        # Note: `log_prob()` already weights the probabilities with the mode weights (if multi-modal) !
        dist_dict = self.env.compute_distributions(ego_trajectory)
        objective = torch.zeros(1)
        for ado_id in ado_ids:
            # p = self._dist_un_conditioned[ado_id].log_prob(dist_dict[ado_id].mean)
            p = dist_dict[ado_id].log_prob(self._dist_un_conditioned[ado_id].mean)
            objective += torch.sum(p)
        objective = objective / len(ado_ids)  # average over ado-ids

        # We want to maximize the probability of the unconditioned trajectories in the conditioned
        # distribution, so we minimize its negative value. When the projected trajectory is very unlikely
        # in the computed distribution, then the log likelihood grows to very large values, which would
        # result in very un-balanced objectives. Therefore it is clamped.
        objective_min = - objective
        max_value = mantrap.constants.OBJECTIVE_PROB_INTERACT_MAX
        return objective_min.clamp(-max_value, max_value)

    def normalize(self, x: typing.Union[np.ndarray, float]) -> typing.Union[np.ndarray, float]:
        """Normalize the objective/constraint value for improved optimization performance.

        The objective value is clamped to some maximal value which enforces the resulting objective
        values to be in some range. And can hence serve as normalize factor.

        :param x: objective/constraint value in normal value range.
        :returns: normalized objective/constraint value in range [-1, 1].
        """
        return x / mantrap.constants.OBJECTIVE_PROB_INTERACT_MAX

    def gradient_condition(self) -> bool:
        """Condition for back-propagating through the objective/constraint in order to obtain the
        objective's gradient vector/jacobian (numerically). If returns True and the ego_trajectory
        itself requires a gradient, the objective/constraint value, stored from the last computation
        (`_current_`-variables) has to require a gradient as well.

        If the internal environment is itself differentiable with respect to the ego (trajectory) input, the
        resulting objective value must have a gradient as well.
        """
        return self.env.is_differentiable_wrt_ego

    ###########################################################################
    # Objective Properties ####################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "interaction_prob"
