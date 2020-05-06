import abc
import logging
import typing

import numpy as np
import torch

import mantrap.environment
import mantrap.utility.shaping


class ConstraintModule(abc.ABC):
    """General constraint class.

    For an unified and general implementation of the constraint function modules this superclass implements methods
    for computing and logging the constraint value as well as the jacobian matrix simply based on a single method,
    the `_compute()` method, which has to be implemented in the child classes. `_compute()` returns the constraint value
    given a planned ego trajectory, while building a torch computation graph, which is used later on to determine the
    jacobian matrix using the PyTorch autograd library.

    :param t_horizon: planning time horizon in number of time-steps (>= 1).
    :param env: environment object reference.
    """
    def __init__(self, t_horizon: int, env: mantrap.environment.base.GraphBasedEnvironment):
        assert t_horizon >= 1
        self._t_horizon = t_horizon
        self._env = env

        # Logging variables for objective and gradient values. For logging the latest variables are stored
        # as class parameters and appended to the log when calling the `logging()` function, in order to avoid
        # appending multiple values within one optimization step.
        self._constraint_current = None

    ###########################################################################
    # Constraint Formulation ##################################################
    ###########################################################################
    def constraint(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str] = None) -> np.ndarray:
        """Determine constraint value for passed ego trajectory by calling the internal `compute()` method.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, self.t_horizon + 1, pos_and_vel_only=True)
        constraints = self._compute(ego_trajectory, ado_ids=ado_ids)
        constraints = constraints.detach().numpy() if constraints is not None else np.array([])
        return self._return_constraint(constraints)

    def jacobian(self, ego_trajectory: torch.Tensor, grad_wrt: torch.Tensor, ado_ids: typing.List[str] = None
                 ) -> np.ndarray:
        """Determine jacobian matrix for passed ego trajectory.

        Therefore at first check whether an analytical solution is defined, if not determine the constraint values
        by calling the internal `compute()` method and en passant build a computation graph. Then using the PyTorch
        autograd library compute the jacobian matrix based on the constraints computation graph.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param grad_wrt: vector w.r.t. which the gradient should be determined.
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, self.t_horizon + 1, pos_and_vel_only=True)

        # Analytical solutions are more exact and (usually more efficient) to compute, when known, compared
        # to the numerical "graphical" solution. Therefore, first check whether an analytical solution is
        # defined for this module.
        jacobian_analytical = self._compute_jacobian_analytically(ego_trajectory, grad_wrt, ado_ids=ado_ids)
        if jacobian_analytical is not None:
            jacobian = jacobian_analytical

        # Otherwise compute the jacobian using torch auto-grad function, for each constraint individually.
        else:
            # Compute the constraint values and check whether a gradient between them and the ego_trajectory
            # input (which has been assured to require a gradient) exists, if the module-conditions for
            # that are met.
            constraints = self._compute(ego_trajectory, ado_ids=ado_ids)

            # If constraint vector is None, directly return empty jacobian vector.
            assert grad_wrt.requires_grad
            assert ego_trajectory.requires_grad  # otherwise constraints cannot have gradient function
            if constraints is None or not self._constraints_gradient_condition():
                jacobian = np.array([])

            # Otherwise check for the existence of a gradient, as explained above.
            # In general the constraints might not be affected by the `ego_trajectory`, then they does not have
            # gradient function and the gradient is not defined. Then the jacobian is assumed to be zero.
            else:
                assert constraints.requires_grad
                grad_size = int(grad_wrt.numel())
                constraint_size = int(constraints.numel())
                if constraint_size == 1:
                    jacobian = torch.autograd.grad(constraints, grad_wrt, retain_graph=True)[0]
                else:
                    jacobian = torch.zeros(constraint_size * grad_size)
                    for i, x in enumerate(constraints):
                        grad = torch.autograd.grad(x, grad_wrt, retain_graph=True)[0]
                        jacobian[i * grad_size:(i + 1) * grad_size] = grad.flatten().detach()
                jacobian = jacobian.detach().numpy()

        return jacobian

    @abc.abstractmethod
    def _compute(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str] = None
                 ) -> typing.Union[torch.Tensor, None]:
        """Determine constraint value core method.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        raise NotImplementedError

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

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param grad_wrt: vector w.r.t. which the gradient should be determined.
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        return None

    @abc.abstractmethod
    def _constraints_gradient_condition(self) -> bool:
        """Determine constraint violation based on the last constraint evaluation.

        The violation is the amount how much the solution state is inside the constraint active region.
        When the constraint is not active, then the violation is zero. The calculation is based on the last (cached)
        evaluation of the constraint function.
        """
        raise NotImplementedError

    ###########################################################################
    # Constraint Bounds #######################################################
    ###########################################################################
    def _constraint_boundaries(self) -> typing.Tuple[typing.Union[float, None], typing.Union[float, None]]:
        """Lower and upper bounds for constraint values."""
        raise NotImplementedError

    def constraint_boundaries(
        self, ado_ids: typing.List[str] = None
    ) -> typing.Tuple[typing.Union[np.ndarray, typing.List[None]], typing.Union[np.ndarray, typing.List[None]]]:
        lower, upper = self._constraint_boundaries()
        num_constraints = self.num_constraints(ado_ids=ado_ids)
        lower = lower * np.ones(num_constraints) if lower is not None else [None] * num_constraints
        upper = upper * np.ones(num_constraints) if upper is not None else [None] * num_constraints
        return lower, upper

    def num_constraints(self, ado_ids: typing.List[str] = None) -> int:
        raise NotImplementedError

    ###########################################################################
    # Constraint Violation ####################################################
    ###########################################################################
    def compute_violation(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str] = None) -> float:
        """Determine constraint violation based on some input ego trajectory and ado ids list.

        The violation is the amount how much the solution state is inside the constraint active region.
        When the constraint is not active, then the violation is zero. The calculation is based on the last
        (cached) evaluation of the constraint function.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        assert mantrap.utility.shaping.check_ego_trajectory(x=ego_trajectory, pos_and_vel_only=True)
        constraint = self._compute(ego_trajectory, ado_ids=ado_ids)
        return self._violation(constraint=constraint.detach().numpy())

    def compute_violation_internal(self) -> float:
        """Determine constraint violation, i.e. how much the internal state is inside the constraint active region.
        When the constraint is not active, then the violation is zero. The calculation is based on the last (cached)
        evaluation of the constraint function.
        """
        return self._violation(constraint=self._constraint_current)

    def _violation(self, constraint: np.ndarray) -> float:
        num_constraints = constraint.size
        no_violation = np.zeros(num_constraints)
        lower, upper = self._constraint_boundaries()
        violation_lower = lower * np.ones(num_constraints) - constraint if lower is not None else no_violation
        violation_upper = constraint - upper * np.ones(num_constraints) if upper is not None else no_violation
        return float(np.sum(np.maximum(no_violation, violation_lower) + np.maximum(no_violation, violation_upper)))

    ###########################################################################
    # Utility #################################################################
    ###########################################################################
    def _return_constraint(self, constraint_value: np.ndarray) -> np.ndarray:
        self._constraint_current = constraint_value
        logging.debug(f"Module {self.__str__()} computed")
        return self._constraint_current

    ###########################################################################
    # Constraint Properties ###################################################
    ###########################################################################
    @property
    def constraint_current(self) -> np.ndarray:
        return self._constraint_current

    @property
    def inf_current(self) -> float:
        """Current infeasibility value (amount of constraint violation)."""
        return self.compute_violation_internal()

    @property
    def t_horizon(self) -> int:
        return self._t_horizon

    @property
    def name(self) -> str:
        raise NotImplementedError
