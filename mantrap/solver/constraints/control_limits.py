from typing import List, Tuple, Union

import torch

from mantrap.solver.constraints.constraint_module import ConstraintModule


class ControlLimitModule(ConstraintModule):
    """Maximal control input at every point in time.

    For computing this constraint simply the norm of the planned control input is determined and compared to the
    maximal agent's control limit. For 0 < t < T_{planning}:

    .. math:: ||u(t)||_2 < u_{max}
    """

    def _compute(self, ego_trajectory: torch.Tensor, ado_ids: List[str] = None) -> Union[torch.Tensor, None]:
        """Determine constraint value core method.

        The max control constraints simply are computed by transforming the given trajectory to control input
        (deterministic dynamics). Then take the L2 norm over the "cartesian" axis to get the norm of the
        control input at every time-step.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        controls = self._env.ego.roll_trajectory(ego_trajectory, dt=self._env.dt)
        return torch.norm(controls, dim=1).flatten()

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
    def _constraint_boundaries(self) -> Tuple[Union[float, None], Union[float, None]]:
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

    def num_constraints(self, ado_ids: List[str] = None) -> int:
        return self.t_horizon
