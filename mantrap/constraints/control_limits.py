import typing

import numpy as np
import torch

import mantrap.constraints
import mantrap.utility.shaping


class ControlLimitModule(mantrap.constraints.ConstraintModule):
    """Maximal control input at every point in time.

    For computing this constraint simply the norm of the planned control input is determined and compared to the
    maximal agent's control limit. For 0 < t < T_{planning}:

    .. math:: ||u(t)||_2 < u_{max}
    """

    def _compute(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str] = None
                 ) -> typing.Union[torch.Tensor, None]:
        """Determine constraint value core method.

        The max control constraints simply are computed by transforming the given trajectory to control input
        (deterministic dynamics). Then take the L2 norm over the "cartesian" axis to get the norm of the
        control input at every time-step.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        ego_controls = self._env.ego.roll_trajectory(ego_trajectory, dt=self._env.dt)
        return torch.norm(ego_controls, dim=1).flatten()

    def _compute_jacobian_analytically(
        self, ego_trajectory: torch.Tensor, grad_wrt: torch.Tensor, ado_ids: typing.List[str] = None
    ) -> typing.Union[np.ndarray, None]:
        """Compute Jacobian matrix analytically.

        While the Jacobian matrix of the constraint can be computed automatically using PyTorch's automatic
        differentiation package there might be an analytic solution, which is when known for sure more
        efficient to compute. Although it is against the convention to use torch representations whenever
        possible, this function returns numpy arrays, since the main jacobian() function has to return
        a numpy array. Hence, not computing based on numpy arrays would just introduce an un-necessary
        `.detach().numpy()`.

        When no analytical solution is defined (or too hard to determine) return None.

        When the gradient shall be computed with respect to the controls, then computing the gradient analytically
        is very straight-forward, by just applying the following formula:

        .. math::

            \\frac{ d ||u|| }{ du_i } = \\frac{u_i}{||u||_2}

        Otherwise the analytical jacobian is hard to compute analytically, for a general `grad_wrt`, then
        return `None` in order to go on with numerical computations.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param grad_wrt: vector w.r.t. which the gradient should be determined.
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory)

        with torch.no_grad():

            # Compute controls from trajectory, if not equal to `grad_wrt` return None.
            ego_controls = self._env.ego.roll_trajectory(ego_trajectory, dt=self._env.dt)
            if not torch.all(torch.isclose(ego_controls, grad_wrt)):
                return None

            # Otherwise compute Jacobian using formula in method's description above.
            t_horizon, u_size = ego_controls.shape
            u_norm = torch.norm(ego_controls, dim=1).flatten().detach().numpy()
            u_norm[u_norm == 0.0] = 1e-6  # remove nan values when computing (1 / u_norm)
            u = ego_controls.flatten().numpy()

            u_norm_stretched = np.repeat(u_norm, t_horizon * u_size)
            u_stretched = np.concatenate([u] * t_horizon)
            jacobian = np.repeat(np.eye(t_horizon), u_size) * 1 / u_norm_stretched * u_stretched
            return jacobian.flatten()

    def _constraints_gradient_condition(self) -> bool:
        """Conditions for the existence of a gradient between the input of the constraint value computation
        (which is the ego_trajectory) and the constraint values itself. If returns True and the ego_trajectory
        itself requires a gradient, the constraint output has to require a gradient as well.

        Since the velocities are part of the given ego_trajectory, the gradient should always exist.
        """
        return True

    ###########################################################################
    # Constraint Bounds #######################################################
    ###########################################################################
    def _constraint_boundaries(self) -> typing.Tuple[typing.Union[float, None], typing.Union[float, None]]:
        """Lower and upper bounds for constraint values.

        The boundaries of this constraint depend on the exact implementation of the agent, however most agents
        are isotropic, so assuming to have equal control boundaries in both cartesian directions and also
        have its lower bound smaller or equal to zero, so that we can simplify the constraint to only have an
        upper bound (since the lower bound zero is anyways given and a L2-norm is semi-positive).
        """
        lower, upper = self._env.ego.control_limits()
        if lower <= 0:
            return None, upper
        else:
            return lower, upper

    def num_constraints(self, ado_ids: typing.List[str] = None) -> int:
        return self.t_horizon
