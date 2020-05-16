import abc
import typing

import numpy as np
import torch

import mantrap.environment


class OptimizationModule(abc.ABC):
    def __init__(self, t_horizon: int,  weight: typing.Union[float, None] = 0.0,
                 env: mantrap.environment.base.GraphBasedEnvironment = None,
                 has_slack: bool = False, slack_weight: float = 0.0):
        """General objective and constraint module.

        For an unified and general implementation of objective and constraint function modules, this superclass
        implements methods for computing both, either analytically or numerically based on the PyTorch autograd
        package. Thereby all objective and constraint computations should be purely based on the robot's (ego)
        trajectory, as well as the possibility to perform further roll-outs in a given simulation environment.

        Combining objective and constraint in one module object grants the possibility to introduce soft
        constraints using slack variables, while also merely implementing pure objective and constraint
        modules seamlessly.
        For making the constraints soft, slack variables are added internally to the objective and updated
        with every constraint call. Therefore only the `has_slack` flag has to be set to True. However for
        simplicity we assume that then each constraint of the defined module is soft, otherwise the module
        can just divided into two modules.

        For multiprocessing the same module object is shared over all processes (even sharing memory), in order to
        avoid repeated pre-computation steps online. However to avoid racing conditions this means that internal
        variables of the class object are also shared and not owned by the process. Altering these variables in
        one process would then lead to un-expected outcomes in the other processes. Therefore each function comes
        with a `tag` argument which classifies the current process the function runs in. When internal variables
        have to be used, then they should be assigned to some dictionary with the tags as keys, so  that the
        function only alters variables which are assigned to this process.
        """
        assert t_horizon is None or t_horizon >= 1
        assert weight is None or weight >= 0.0

        self._weight = weight
        self._t_horizon = t_horizon

        # Giving every optimization module access to a (large) simulation environment object is not
        # necessary and an un-necessary use of space, even when it is just a pointer.
        self._env = env  # may be None

        # Logging variables for objective and gradient values. For logging the latest variables are stored
        # as class parameters and appended to the log when calling the `logging()` function, in order to avoid
        # appending multiple values within one optimization step.
        self._constraint_current = {}  # type: typing.Dict[str, np.ndarray]
        self._obj_current = {}  # type: typing.Dict[str, float]
        self._grad_current = {}  # type: typing.Dict[str, np.ndarray]

        # Slack variables - Slack variables are part of both the constraints and the objective function,
        # therefore have to stored internally to be shared between both functions. However as discussed
        # above during multi-processing the same module object is shared over multiple processes,
        # therefore store the slack variable values in dictionaries assigned to the processes tag.
        assert slack_weight >= 0.0
        self._slack = {}  # type: typing.Dict[str, torch.Tensor]
        self._has_slack = has_slack
        self._slack_weight = slack_weight

    def reset_env(self, env: mantrap.environment.base.GraphBasedEnvironment):
        if self._env is not None:
            self._env = env

    ###########################################################################
    # Objective ###############################################################
    ###########################################################################
    def objective(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str], tag: str) -> float:
        """Determine objective value for passed ego trajectory by calling the internal `compute()` method.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        :param tag: name of optimization call (name of the core).
        """
        obj_value = self.compute_objective(ego_trajectory, ado_ids=ado_ids, tag=tag)
        # Convert objective in standard optimization format (as float).
        if obj_value is None:
            obj_value = 0.0  # if objective not defined simply return 0.0
        else:
            obj_value = float(obj_value.item())
        return self._return_objective(obj_value, tag=tag)

    def compute_objective(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str], tag: str
                          ) -> typing.Union[torch.Tensor, None]:
        """Determine internal objective value + slack variables.
        
        Add slack based part of objective function. The value of the slack variable can only be
        updated if the constraints have been computed before. However using general optimization
        frameworks we cannot enforce the order to method calls, therefore to be surely synched
        we have to compute the constraints here first (!).

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        :param tag: name of optimization call (name of the core).
        """
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, pos_and_vel_only=True)
        obj_value = self._compute_objective(ego_trajectory, ado_ids=ado_ids, tag=tag)

        if self._has_slack:
            obj_value = torch.zeros(1) if obj_value is None else obj_value
            _ = self.compute_constraint(ego_trajectory, ado_ids=ado_ids, tag=tag)
            obj_value += self._slack_weight * self._slack[tag].sum()

        return obj_value

    @abc.abstractmethod
    def _compute_objective(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str], tag: str
                           ) -> typing.Union[torch.Tensor, None]:
        """Determine objective value core method.

        The objective value should be returned either as PyTorch tensor or `None`. It cannot be simplified as
        floating point number directly, as next to its value it is important to return the gradient function,
        when computing its gradient. When the objective is not defined, simply return `None`.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        :param tag: name of optimization call (name of the core).
        """
        raise NotImplementedError

    ###########################################################################
    # Gradient ################################################################
    ###########################################################################
    def gradient(self, ego_trajectory: torch.Tensor, grad_wrt: torch.Tensor, ado_ids: typing.List[str], tag: str
                 ) -> np.ndarray:
        """Determine gradient vector for passed ego trajectory. Therefore determine the objective value by
        calling the internal `compute()` method and en passant build a computation graph. Then using the pytorch
        autograd library compute the gradient vector through the previously built computation graph.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param grad_wrt: vector w.r.t. which the gradient should be determined.
        :param ado_ids: ghost ids which should be taken into account for computation.
        :param tag: name of optimization call (name of the core).
        """
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, pos_and_vel_only=True)

        # Analytical solutions are more exact and (usually more efficient) to compute, when known, compared
        # to the numerical "graphical" solution. Therefore, first check whether an analytical solution is
        # defined for this module.
        gradient_analytical = self._compute_gradient_analytically(ego_trajectory, grad_wrt, ado_ids=ado_ids, tag=tag)
        if gradient_analytical is not None:
            gradient = gradient_analytical

        # Otherwise compute the jacobian using torch auto-grad function, for each constraint individually.
        else:
            assert grad_wrt.requires_grad
            assert ego_trajectory.requires_grad  # otherwise objective cannot have gradient function

            # Compute the objective value and check whether a gradient between the value and the
            # ego_trajectory input (which has been assured to require a gradient) exists, if the
            # module-conditions for that are met.
            objective = self._compute_objective(ego_trajectory, ado_ids=ado_ids, tag=tag)

            # If objective is None return an zero gradient of the length of the `grad_wrt` tensor.
            # In general the objective might not be affected by the `ego_trajectory`, then it does not have
            # a gradient function and the gradient is not defined. Then the objective gradient is assumed
            # to be zero.
            if objective is None or not self._gradient_condition():
                gradient = np.zeros(grad_wrt.numel())

            # Otherwise compute the gradient "numerically" using the PyTorch autograd package.
            else:
                gradient = self._compute_gradient_autograd(objective, grad_wrt=grad_wrt)

        return self._return_gradient(gradient, tag=tag)

    def _compute_gradient_analytically(
        self, ego_trajectory: torch.Tensor, grad_wrt: torch.Tensor, ado_ids: typing.List[str], tag: str
    ) -> typing.Union[np.ndarray, None]:
        """Compute objective gradient vector analytically.

        While the gradient vector of the objective can be computed automatically using PyTorch's automatic
        differentiation package there might be an analytic solution, which is when known for sure more
        efficient to compute. Although it is against the convention to use torch representations whenever
        possible, this function returns numpy arrays, since the main gradient() function has to return
        a numpy array. Hence, not computing based on numpy arrays would just introduce an un-necessary
        `.detach().numpy()`.

        When no analytical solution is defined (or too hard to determine) return None.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param grad_wrt: vector w.r.t. which the gradient should be determined.
        :param ado_ids: ghost ids which should be taken into account for computation.
        :param tag: name of optimization call (name of the core).
        """
        return None

    ###########################################################################
    # Constraint ##############################################################
    ###########################################################################
    def constraint(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str], tag: str) -> np.ndarray:
        """Determine constraint value for passed ego trajectory by calling the internal `compute()` method.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        :param tag: name of optimization call (name of the core).
        """
        constraints = self.compute_constraint(ego_trajectory, ado_ids=ado_ids, tag=tag)
        # Convert constraints in standard optimization format (as numpy arrays).
        if constraints is None:
            constraints = np.array([])
        else:
            constraints = constraints.detach().numpy()
        return self._return_constraint(constraints, tag=tag)

    def compute_constraint(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str], tag: str
                           ) -> typing.Union[torch.Tensor, None]:
        """Determine internal constraints + slack constraints.

        Compute internal constraints and convert them to equality constraints by updating and adding the
        slack variables. Then add further constraints for the slack variables themselves (>= 0).

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        :param tag: name of optimization call (name of the core).
        """
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, pos_and_vel_only=True)
        constraints = self._compute_constraint(ego_trajectory, ado_ids=ado_ids, tag=tag)

        # Update slack variables (if any are defined for this module).
        if self._has_slack and constraints is not None:
            self._slack[tag] = - constraints
            constraints = constraints + self._slack[tag]  # constraint - slack (slacked variables)
            constraints = torch.cat((constraints, self._slack[tag]))

        return constraints

    @abc.abstractmethod
    def _compute_constraint(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str], tag: str
                            ) -> typing.Union[torch.Tensor, None]:
        """Determine constraint value core method.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        :param tag: name of optimization call (name of the core).
        """
        raise NotImplementedError

    ###########################################################################
    # Jacobian ################################################################
    ###########################################################################
    def jacobian(self, ego_trajectory: torch.Tensor, grad_wrt: torch.Tensor, ado_ids: typing.List[str], tag: str
                 ) -> np.ndarray:
        """Determine jacobian matrix for passed ego trajectory.

        Therefore at first check whether an analytical solution is defined, if not determine the constraint values
        by calling the internal `compute()` method and en passant build a computation graph. Then using the PyTorch
        autograd library compute the jacobian matrix based on the constraints computation graph.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param grad_wrt: vector w.r.t. which the gradient should be determined.
        :param ado_ids: ghost ids which should be taken into account for computation.
        :param tag: name of optimization call (name of the core).
        """
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, pos_and_vel_only=True)

        # Analytical solutions are more exact and (usually more efficient) to compute, when known, compared
        # to the numerical "graphical" solution. Therefore, first check whether an analytical solution is
        # defined for this module.
        jacobian_analytical = self._compute_jacobian_analytically(ego_trajectory, grad_wrt, ado_ids=ado_ids, tag=tag)
        if jacobian_analytical is not None:
            jacobian = jacobian_analytical

        # Otherwise compute the jacobian using torch auto-grad function, for each constraint individually.
        else:
            # Compute the constraint values and check whether a gradient between them and the ego_trajectory
            # input (which has been assured to require a gradient) exists, if the module-conditions for
            # that are met.
            assert ego_trajectory.requires_grad  # otherwise constraints cannot have gradient function
            constraints = self._compute_constraint(ego_trajectory, ado_ids=ado_ids, tag=tag)

            # If constraint vector is None, directly return empty jacobian vector.
            if constraints is None:
                jacobian = np.array([])

            # Otherwise check for the existence of a gradient, as explained above.
            # In general the constraints might not be affected by the `ego_trajectory`, then they does not have
            # gradient function and the gradient is not defined. Then the jacobian is assumed to be zero.
            else:
                # If constraints are not None (exist) but the gradient cannot be computed, e.g. since the
                # constraints do not depend on the ego_trajectory, then return a zero jacobian.
                if not self._gradient_condition():
                    grad_size = int(grad_wrt.numel())
                    constraint_size = int(constraints.numel())
                    jacobian = np.zeros(grad_size * constraint_size)

                # Otherwise determine the jacobian numerically using the PyTorch autograd package.
                else:
                    jacobian = self._compute_gradient_autograd(constraints, grad_wrt=grad_wrt)

        return jacobian

    def _compute_jacobian_analytically(
        self, ego_trajectory: torch.Tensor, grad_wrt: torch.Tensor, ado_ids: typing.List[str], tag: str
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
        :param tag: name of optimization call (name of the core).
        """
        return None

    ###########################################################################
    # Autograd Differentiation ################################################
    ###########################################################################
    @staticmethod
    def _compute_gradient_autograd(x: torch.Tensor, grad_wrt: torch.Tensor) -> np.ndarray:
        """Compute derivative of x with respect to grad_wrt.

        Compute the gradient/jacobian/etc. of some vector x with respect to some tensor `grad_wrt`
        using the PyTorch autograd, automatic differentiation package. Here we assume that both are
        connected by some computational graph (PyTorch graph) that can be used for differentiation.

        A comment about multiprocessing: Computing the gradients in parallel would be a good match
        for multiple processing, since it is fairly independent from each other, given the shared
        memory of the computation graph.

        ```
        import torch.multiprocessing as mp

        mp.set_start_method('spawn')
        x.share_memory_()
        grad_wrt.share_memory_()
        gradient.share_memory_()

        def compute(x_i, grad_wrt_i):
            grad = torch.autograd.grad(element, grad_wrt, retain_graph=True, only_inputs=True)[0]
            return grad.flatten().detach()

        processes = []
        for i_process in range(8):
            p = mp.Process(target=compute, args=(x[i_process], grad_wrt, ))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        ```

        Here the torch.multiprocessing library is used to compute the gradient over the whole tensor x in
        multiple parallel processes. Therefore the tensors of both x and grad_wrt are shared over all
        processes using the `.share_memory()` method and all processes are launched with a different
        element of the tensor x. However as shown below sharing a computation graph, i.e. tensors that
        require a gradient, being attached to this graph, over multiple processes is not supported in
        PyTorch and therefore not possible.

        ```
        def reduce_tensor(tensor):
            storage = tensor.storage()

            if tensor.requires_grad and not tensor.is_leaf:
                raise RuntimeError("Cowardly refusing to serialize non-leaf tensor which requires_grad, "
                                   "since autograd does not support crossing process boundaries.  "
                                   "If you just want to transfer the data, call detach() on the tensor "
                                   "before serializing (e.g., putting it on the queue).")
                RuntimeError: Cowardly refusing to serialize non-leaf tensor which requires_grad, since
                autograd does not support crossing process boundaries.  If you just want to transfer the data,
                call detach() on the tensor before serializing (e.g., putting it on the queue).
        ```

        To avoid this issue, the full computation graph would have to be re-built for every single element
        of x, which would create a lot of overhead due to repeated computations (as well as being quite not
        general and unreadable due to nesting instead of batching) and therefore not accelerate the computations.

        :params x: gradient input flat vector.
        :param grad_wrt: tensor with respect to gradients should be computed.
        :returns: flattened gradient tensor (x.size * grad_wrt.size)
        """
        grad_size = int(grad_wrt.numel())
        x_size = int(x.numel())
        assert x.requires_grad
        assert grad_wrt.requires_grad

        # Compute gradient batched, i.e. per element of x over the full `grad_wrt` tensor. However further
        # batching unfortunately is not possible using the autograd framework.
        if x_size == 1:
            gradient = torch.autograd.grad(x, grad_wrt, retain_graph=True)[0]
        else:
            gradient = torch.zeros(x_size * grad_size)
            for i, element in enumerate(x):
                grad = torch.autograd.grad(element, grad_wrt, retain_graph=True)[0]
                gradient[i * grad_size:(i + 1) * grad_size] = grad.flatten().detach()

        return gradient.flatten().detach().numpy()

    ###########################################################################
    # Constraint Bounds #######################################################
    ###########################################################################
    def constraint_boundaries(
        self, ado_ids: typing.List[str] = None
    ) -> typing.Tuple[typing.Union[typing.List[float], typing.List[None]],
                      typing.Union[typing.List[float], typing.List[None]]]:
        # Module-individual constraint boundaries.
        lower, upper = self._constraint_boundaries()
        num_constraints = self.num_constraints(ado_ids=ado_ids)
        lower_bounds = (lower * np.ones(num_constraints)).tolist() \
            if lower is not None else [None] * num_constraints
        upper_bounds = (upper * np.ones(num_constraints)).tolist() \
            if upper is not None else [None] * num_constraints

        # Slack variable introduced boundaries. We assume that the number of slack variables
        # is equal to number of constraint, i.e. that each constraint of the module is "soft".
        if self._has_slack:
            lower_bounds = lower_bounds + [0.0] * num_constraints
            upper_bounds = upper_bounds + [None] * num_constraints
        return lower_bounds, upper_bounds

    def _constraint_boundaries(self) -> typing.Tuple[typing.Union[float, None], typing.Union[float, None]]:
        """Lower and upper bounds for constraint values."""
        raise NotImplementedError

    def num_constraints(self, ado_ids: typing.List[str]) -> int:
        num_module_constraints = self._num_constraints(ado_ids=ado_ids)
        num_slack_constraints = num_module_constraints if self._has_slack else 0
        return num_module_constraints + num_slack_constraints

    @abc.abstractmethod
    def _num_constraints(self, ado_ids: typing.List[str]) -> int:
        raise NotImplementedError

    ###########################################################################
    # Constraint Violation ####################################################
    ###########################################################################
    def compute_violation(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str], tag: str) -> float:
        """Determine constraint violation based on some input ego trajectory and ado ids list.

        The violation is the amount how much the solution state is inside the constraint active region.
        When the constraint is not active, then the violation is zero. The calculation is based on the last
        (cached) evaluation of the constraint function.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        :param tag: name of optimization call (name of the core).
        """
        assert mantrap.utility.shaping.check_ego_trajectory(x=ego_trajectory, pos_and_vel_only=True)
        constraint = self._compute_constraint(ego_trajectory, ado_ids=ado_ids, tag=tag)
        if constraint is None:
            return self._violation(constraint=None)
        else:
            return self._violation(constraint=constraint.detach().numpy())

    def compute_violation_internal(self, tag: str) -> float:
        """Determine constraint violation, i.e. how much the internal state is inside the constraint active region.
        When the constraint is not active, then the violation is zero. The calculation is based on the last (cached)
        evaluation of the constraint function.
        """
        return self._violation(constraint=self._constraint_current[tag])

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

    def _return_constraint(self, constraint_value: np.ndarray, tag: str) -> np.ndarray:
        self._constraint_current[tag] = constraint_value
        return self._constraint_current[tag]

    def _return_objective(self, obj_value: float, tag: str) -> float:
        self._obj_current[tag] = self.weight * obj_value
        return self._obj_current[tag]

    def _return_gradient(self, gradient: np.ndarray, tag: str) -> np.ndarray:
        self._grad_current[tag] = self.weight * gradient
        return self._grad_current[tag]

    ###########################################################################
    # Module backlog ##########################################################
    ###########################################################################
    def constraint_current(self, tag: str) -> np.ndarray:
        return self._constraint_current[tag]

    def inf_current(self, tag: str) -> float:
        return self.compute_violation_internal(tag=tag)

    def obj_current(self, tag: str) -> float:
        return self._obj_current[tag]

    def grad_current(self, tag: str) -> float:
        return np.linalg.norm(self._grad_current[tag])

    def slack_variables(self, tag: str) -> torch.Tensor:
        return self._slack[tag]

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
