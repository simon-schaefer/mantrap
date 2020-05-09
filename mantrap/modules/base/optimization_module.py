import abc
import typing

import numpy as np
import torch

import mantrap.environment


class OptimizationModule(abc.ABC):
    def __init__(self, t_horizon: int,  weight: float = 0.0, **module_kwargs):
        """General objective and constraint module.

        For an unified and general implementation of objective and constraint function modules, this superclass
        implements methods for computing both, either analytically or numerically based on the PyTorch autograd
        package. Thereby all objective and constraint computations should be purely based on the robot's (ego)
        trajectory, as well as the possibility to perform further roll-outs in a given simulation environment.

        Combining objective and constraint in one module object grants the possibility to introduce soft
        constraints using slack variables, while also merely implementing pure objective and constraint
        modules seamlessly.
        """
        assert t_horizon >= 1
        assert weight >= 0.0

        self._weight = weight
        self._t_horizon = t_horizon

        # Giving every optimization module access to a (large) simulation environment object is not
        # necessary and an un-necessary use of space, even when it is just a pointer. Therefore using
        # the simulation requires executing another class method (`initialize_env`).
        self._env = None

        # Logging variables for objective and gradient values. For logging the latest variables are stored
        # as class parameters and appended to the log when calling the `logging()` function, in order to avoid
        # appending multiple values within one optimization step.
        self._constraint_current = np.array([])
        self._obj_current = 0.0
        self._grad_current = None

    def initialize_env(self, env: mantrap.environment.base.GraphBasedEnvironment):
        assert env.ego is not None
        self._env = env

    def reset_env(self, env: mantrap.environment.base.GraphBasedEnvironment):
        if self._env is not None:
            self.initialize_env(env=env)

    ###########################################################################
    # Objective ###############################################################
    ###########################################################################
    def objective(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str] = None) -> float:
        """Determine objective value for passed ego trajectory by calling the internal `compute()` method.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, pos_and_vel_only=True)
        obj_value = self._compute_objective(ego_trajectory, ado_ids=ado_ids)
        if obj_value is None:
            obj_value = 0.0  # if objective not defined simply return 0.0
        else:
            obj_value = float(obj_value.item())
        return self._return_objective(obj_value)

    @abc.abstractmethod
    def _compute_objective(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str] = None
                           ) -> typing.Union[torch.Tensor, None]:
        """Determine objective value core method.

        The objective value should be returned either as PyTorch tensor or `None`. It cannot be simplified as
        floating point number directly, as next to its value it is important to return the gradient function,
        when computing its gradient. When the objective is not defined, simply return `None`.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        raise NotImplementedError

    ###########################################################################
    # Gradient ################################################################
    ###########################################################################
    def gradient(self, ego_trajectory: torch.Tensor, grad_wrt: torch.Tensor, ado_ids: typing.List[str] = None
                 ) -> np.ndarray:
        """Determine gradient vector for passed ego trajectory. Therefore determine the objective value by
        calling the internal `compute()` method and en passant build a computation graph. Then using the pytorch
        autograd library compute the gradient vector through the previously built computation graph.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param grad_wrt: vector w.r.t. which the gradient should be determined.
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, pos_and_vel_only=True)
        assert grad_wrt.requires_grad
        assert ego_trajectory.requires_grad  # otherwise objective cannot have gradient function

        # Compute the objective value and check whether a gradient between the value and the ego_trajectory input
        # (which has been assured to require a gradient) exists, if the module-conditions for that are met.
        objective = self._compute_objective(ego_trajectory, ado_ids=ado_ids)

        # If objective is None return an zero gradient of the length of the `grad_wrt` tensor.
        if objective is None or not self._gradient_condition():
            gradient = np.zeros(grad_wrt.numel())

        # Otherwise check for the existence of a gradient, as explained above.
        # In general the objective might not be affected by the `ego_trajectory`, then it does not have a gradient
        # function and the gradient is not defined. Then the objective gradient is assumed to be zero.
        else:
            assert objective.requires_grad
            gradient = torch.autograd.grad(objective, grad_wrt, retain_graph=True, allow_unused=False)[0]
            gradient = gradient.flatten().detach().numpy()

        return self._return_gradient(gradient)

    ###########################################################################
    # Constraint ##############################################################
    ###########################################################################
    def constraint(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str] = None) -> np.ndarray:
        """Determine constraint value for passed ego trajectory by calling the internal `compute()` method.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, pos_and_vel_only=True)
        constraints = self._compute_constraint(ego_trajectory, ado_ids=ado_ids)
        if constraints is None:
            constraints = np.array([])
        else:
            constraints = constraints.detach().numpy()
        return self._return_constraint(constraints)

    @abc.abstractmethod
    def _compute_constraint(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str] = None
                            ) -> typing.Union[torch.Tensor, None]:
        """Determine constraint value core method.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        raise NotImplementedError

    ###########################################################################
    # Jacobian ################################################################
    ###########################################################################
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
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, pos_and_vel_only=True)

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
            constraints = self._compute_constraint(ego_trajectory, ado_ids=ado_ids)

            # If constraint vector is None, directly return empty jacobian vector.
            assert grad_wrt.requires_grad
            assert ego_trajectory.requires_grad  # otherwise constraints cannot have gradient function
            if constraints is None:
                jacobian = np.array([])

            # Otherwise check for the existence of a gradient, as explained above.
            # In general the constraints might not be affected by the `ego_trajectory`, then they does not have
            # gradient function and the gradient is not defined. Then the jacobian is assumed to be zero.
            else:
                grad_size = int(grad_wrt.numel())
                constraint_size = int(constraints.numel())

                # If constraints are not None (exist) but the gradient cannot be computed, e.g. since the
                # constraints do not depend on the ego_trajectory, then return a zero jacobian.
                if not self._gradient_condition():
                    jacobian = np.zeros(grad_size * constraint_size)

                # Otherwise determine the jacobian numerically using the PyTorch autograd package.
                else:
                    assert constraints.requires_grad
                    if constraint_size == 1:
                        jacobian = torch.autograd.grad(constraints, grad_wrt, retain_graph=True)[0]
                    else:
                        jacobian = torch.zeros(constraint_size * grad_size)
                        for i, x in enumerate(constraints):
                            grad = torch.autograd.grad(x, grad_wrt, retain_graph=True)[0]
                            jacobian[i * grad_size:(i + 1) * grad_size] = grad.flatten().detach()
                    jacobian = jacobian.flatten().detach().numpy()

        return jacobian

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
        constraint = self._compute_constraint(ego_trajectory, ado_ids=ado_ids)
        if constraint is None:
            return self._violation(constraint=None)
        else:
            return self._violation(constraint=constraint.detach().numpy())

    def compute_violation_internal(self) -> float:
        """Determine constraint violation, i.e. how much the internal state is inside the constraint active region.
        When the constraint is not active, then the violation is zero. The calculation is based on the last (cached)
        evaluation of the constraint function.
        """
        return self._violation(constraint=self._constraint_current)

    def _violation(self, constraint: typing.Union[np.ndarray, None]) -> float:
        if constraint is None:
            return 0.0

        num_constraints = constraint.size
        no_violation = np.zeros(num_constraints)
        lower, upper = self._constraint_boundaries()
        violation_lower = lower * np.ones(num_constraints) - constraint if lower is not None else no_violation
        violation_upper = constraint - upper * np.ones(num_constraints) if upper is not None else no_violation

        # Due to numerical (precision) errors the violation might be non-zero, although the derived optimization
        # variable is just at the constraint border (as for example in linear programming). Ignore these violations.
        violation = np.sum(np.maximum(no_violation, violation_lower) + np.maximum(no_violation, violation_upper))
        if np.abs(violation) < mantrap.constants.CONSTRAINT_VIOLATION_PRECISION:
            return 0.0
        else:
            return float(violation)

    ###########################################################################
    # Utility #################################################################
    ###########################################################################
    @abc.abstractmethod
    def _gradient_condition(self) -> bool:
        """Condition for back-propagating through the objective/constraint in order to obtain the
        objective's gradient vector/jacobian (numerically). If returns True and the ego_trajectory
        itself requires a gradient, the objective/constraint value, stored from the last computation
        (`_current_`-variables) has to require a gradient as well."""
        raise NotImplementedError

    def _return_constraint(self, constraint_value: np.ndarray) -> np.ndarray:
        self._constraint_current = constraint_value
        return self._constraint_current

    def _return_objective(self, obj_value: float) -> float:
        self._obj_current = self.weight * obj_value
        return self._obj_current

    def _return_gradient(self, gradient: np.ndarray) -> np.ndarray:
        self._grad_current = self.weight * gradient
        return self._grad_current

    ###########################################################################
    # Module backlog ##########################################################
    ###########################################################################
    @property
    def constraint_current(self) -> np.ndarray:
        return self._constraint_current

    @property
    def inf_current(self) -> float:
        return self.compute_violation_internal()

    @property
    def obj_current(self) -> float:
        return self._obj_current

    @property
    def grad_current(self) -> float:
        return np.linalg.norm(self._grad_current)

    ###########################################################################
    # Module Properties #######################################################
    ###########################################################################
    @property
    def weight(self) -> float:
        return self._weight

    @property
    def t_horizon(self) -> int:
        return self._t_horizon

    @property
    def name(self) -> str:
        raise NotImplementedError
