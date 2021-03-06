import typing

import numpy as np
import torch

import mantrap.utility.maths
import mantrap.utility.shaping

from .base import PureConstraintModule


class ControlLimitModule(PureConstraintModule):
    """Maximal control input at every point in time.

    For computing this constraint simply the norm of the planned control input is determined and compared to the
    maximal agent's control limit. For 0 < t < T_{planning}:

    .. math:: ||u(t)|| < u_{max}
    """
    def __init__(self, env: mantrap.environment.base.GraphBasedEnvironment, t_horizon: int, **unused):
        super(ControlLimitModule, self).__init__(env=env, t_horizon=t_horizon)

    def _constraint_core(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str], tag: str
                         ) -> typing.Union[torch.Tensor, None]:
        """Determine constraint value core method.

        The max control constraints simply are computed by transforming the given trajectory to control input
        (deterministic dynamics). Then take the norm over the "cartesian" axis to get the norm of the
        control input at every time-step.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        :param tag: name of optimization call (name of the core).
        """
        ego_controls = self.env.ego.roll_trajectory(ego_trajectory, dt=self.env.dt)
        return self.env.ego.control_norm(controls=ego_controls).flatten()

    def compute_jacobian_analytically(
        self, ego_trajectory: torch.Tensor, grad_wrt: torch.Tensor, ado_ids: typing.List[str], tag: str
    ) -> typing.Union[np.ndarray, None]:
        """Compute Jacobian matrix analytically.

        While the Jacobian matrix of the constraint can be computed automatically using PyTorch's automatic
        differentiation package there might be an analytic solution, which is when known for sure more
        efficient to compute. Although it is against the convention to use torch representations whenever
        possible, this function returns numpy arrays, since the main jacobian() function has to return
        a numpy array. Hence, not computing based on numpy arrays would just introduce an un-necessary
        `.detach().numpy()`.

        When the gradient shall be computed with respect to the controls, then computing the gradient analytically
        is very straight-forward, by just applying the following formula:

        .. math:: \\frac{ d ||u|| }{ du_i } = \\frac{u_i}{||u||}

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param grad_wrt: vector w.r.t. which the gradient should be determined.
        :param ado_ids: ghost ids which should be taken into account for computation.
        :param tag: name of optimization call (name of the core).
        """
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory)

        with torch.no_grad():
            t_horizon, u_size = self.t_horizon,  2
            jacobian = np.zeros((u_size * t_horizon, u_size * t_horizon))
            for t in range(t_horizon):
                jacobian[u_size * t, u_size * t] = 1
                jacobian[u_size * t + 1, u_size * t + 1] = 1
            return jacobian.flatten()

    def jacobian_structure(self, ado_ids: typing.List[str], tag: str) -> typing.Union[np.ndarray, None]:
        """Return the sparsity structure of the jacobian, i.e. the indices of non-zero elements.

        :param ado_ids: ghost ids which should be taken into account for computation.
        :param tag: name of optimization call (name of the core).
        :returns: indices of non-zero elements of jacobian.
        """
        t_horizon, u_size = self.t_horizon, 2
        non_zero_indices = [(u_size * t_horizon + 1) * n for n in range(2 * t_horizon)]
        return np.array(non_zero_indices)

    def normalize(self, x: typing.Union[np.ndarray, float]) -> typing.Union[np.ndarray, float]:
        """Normalize the objective/constraint value for improved optimization performance.

        Use the maximally allowed controls as normalize factor, which are a property of the
        robot itself, that is stored in the internal environment representation.

        :param x: objective/constraint value in normal value range.
        :returns: normalized objective/constraint value in range [0, 1].
        """
        _, upper = self.env.ego.control_limits()
        return x / upper

    def gradient_condition(self) -> bool:
        """Condition for back-propagating through the objective/constraint in order to obtain the
        objective's gradient vector/jacobian (numerically). If returns True and the ego_trajectory
        itself requires a gradient, the objective/constraint value, stored from the last computation
        (`_current_`-variables) has to require a gradient as well.

        Since the ego trajectory directly depends on the controls, the gradient always exists.
        """
        return True

    ###########################################################################
    # Constraint Bounds #######################################################
    ###########################################################################
    def _constraint_limits(self) -> typing.Tuple[typing.Union[float, None], typing.Union[float, None]]:
        """Lower and upper bounds for constraint values.

        The boundaries of this constraint depend on the exact implementation of the agent, however most agents
        are isotropic, so assuming to have equal control boundaries in both cartesian directions and also
        have its lower bound smaller or equal to zero, so that we can simplify the constraint to only have an
        upper bound (since the lower bound zero is anyways given and a norm is semi-positive).
        """
        _, control_max = self.env.ego.control_limits()
        return None, control_max

    def _num_constraints(self, ado_ids: typing.List[str]) -> int:
        return self.t_horizon * 2

    ###########################################################################
    # Constraint Properties ###################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "control_limits"
