import typing

import numpy as np
import torch
import torch.distributions

import mantrap.constants
import mantrap.environment
import mantrap.utility

from ..base import PureConstraintModule


class InteractionEllipsoidModule(PureConstraintModule):
    """Constraint based on ellipsoidal constraints around each pedestrian.

    The ellipsoid is described as the prediction probability level set of the N most important
    (i.e. most probable) modes of each pedestrian. Since the prediction is given as a GMM, the
    modes have Gaussian shape and are therefore naturally in elliptical shape.

    .. math:: (rx - \\mu_x)^2 / \\sigma_x^2 + (ry - \\mu_y)^2 / \\sigma_y^2 >= 1

    """

    def __init__(self, env: mantrap.environment.base.GraphBasedEnvironment, t_horizon: int, **unused):
        super(InteractionEllipsoidModule, self).__init__(env=env, t_horizon=t_horizon)

    def _constraint_core(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str], tag: str
                         ) -> typing.Union[torch.Tensor, None]:
        """Determine constraint value core method.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        :param tag: name of optimization call (name of the core).
        """
        if len(ado_ids) == 0 or self.env.num_ados == 0:
            return None

        # Compute the prediction distributions for each pedestrian.
        pos_dist_dict = self.env.compute_distributions(ego_trajectory, vel_dist=False)
        num_most_imp_modes = min(mantrap.constants.CONSTRAINT_ELLIPSOID_NUM_MODES, self.env.num_modes)
        pos_means, pos_sigmas = self.summarize_distribution(pos_dist_dict, num_modes=num_most_imp_modes)

        # Define ellipsoidal constraints as described above.
        constraints = torch.zeros((self.env.num_ados, self.t_horizon, num_most_imp_modes))
        for m_ado in range(self.env.num_ados):
            for t in range(self.t_horizon):
                for m_mode in range(num_most_imp_modes):

                    # For numerical stability threshold the minimal size of the ellipsoid.
                    ellipsoid_dxy = pos_sigmas[m_ado, t, m_mode, :].clamp(min=0.01)
                    dxy = ego_trajectory[t, 0:2] - pos_means[m_ado, t, m_mode, :]

                    ellipsoid = torch.div(dxy.pow(2), ellipsoid_dxy).sum()
                    constraints[m_ado, t, m_mode] = ellipsoid

        return constraints.flatten()

    def summarize_distribution(self, pos_dist_dict: typing.Dict[str, torch.distributions.Distribution],
                               num_modes: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Compute ado-wise positions from positional distribution dict mean and std values."""
        _, ado_states = self.env.states()
        pos_means = torch.zeros((self.env.num_ados, self.t_horizon + 1, num_modes, 2))
        pos_sigmas = torch.zeros((self.env.num_ados, self.t_horizon + 1, num_modes, 2))

        for ado_id, distribution in pos_dist_dict.items():
            m_ado = self.env.index_ado_id(ado_id)

            if self.env.num_modes > num_modes:
                modes_mi = distribution.modes_sorted()[:, -num_modes:]  # most-important modes
            else:
                modes_mi = np.stack([np.arange(0, self.env.num_modes)] * self.t_horizon)

            for t in range(self.t_horizon):
                pos_means[m_ado, t, :, :] = distribution.mean[t, modes_mi[t, :], :].detach()
                pos_sigmas[m_ado, t, :, :] = distribution.stddev[t, modes_mi[t, :], :].detach()

        return pos_means, pos_sigmas

    def gradient_condition(self) -> bool:
        """Condition for back-propagating through the objective/constraint in order to obtain the
        objective's gradient vector/jacobian (numerically). If returns True and the ego_trajectory
        itself requires a gradient, the objective/constraint value, stored from the last computation
        (`_current_`-variables) has to require a gradient as well.

        If the internal environment is itself differentiable with respect to the ego (trajectory) input, the
        resulting objective value must have a gradient as well.
        """
        return self._env.is_differentiable_wrt_ego

    def normalize(self, x: typing.Union[np.ndarray, float]) -> typing.Union[np.ndarray, float]:
        """Normalize the objective/constraint value for improved optimization performance.

        The ellipsoidal constraint intrinsically is normalized, therefore no further normalization
        is required.

        :param x: objective/constraint value in normal value range.
        :returns: normalized objective/constraint value in range [0, 1].
        """
        return x

    ###########################################################################
    # Constraint Bounds #######################################################
    ###########################################################################
    def _constraint_limits(self) -> typing.Tuple[typing.Union[float, None], typing.Union[float, None]]:
        """Lower and upper bounds for constraint values.

        The constraint basically is a lower bound for the distance between the robot and the pedestrian
        at a certain point in time. Therefore, there is no upper bound while the lower bound is one
        (= being on the ellipse border line).
        """
        return 1, None

    def _num_constraints(self, ado_ids: typing.List[str]) -> int:
        num_modes_im = min(mantrap.constants.CONSTRAINT_ELLIPSOID_NUM_MODES, self.env.num_modes)
        return num_modes_im * len(ado_ids) * self.t_horizon

    ###########################################################################
    # Constraint Properties ###################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "ellipsoid_interaction"
