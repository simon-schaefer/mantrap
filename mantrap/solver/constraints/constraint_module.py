from abc import ABC, abstractmethod
import logging
from typing import List, Tuple, Union

import numpy as np
import torch

from mantrap.utility.shaping import check_ego_trajectory


class ConstraintModule(ABC):
    """General constraint class.

    For an unified and general implementation of the constraint function modules this superclass implements methods
    for computing and logging the constraint value as well as the jacobian matrix simply based on a single method,
    the `_compute()` method, which has to be implemented in the child classes. `_compute()` returns the constraint value
    given a planned ego trajectory, while building a torch computation graph, which is used later on to determine the
    jacobian matrix using the PyTorch autograd library.

    :param horizon: planning time horizon in number of time-steps (>= 1).
    """
    def __init__(self, horizon: int, **module_kwargs):
        self.T = horizon

        # Initialize module.
        self.initialize(**module_kwargs)

        # Determine constraint bounds (defined in child classes).
        self.lower, self.upper = self.constraint_bounds()
        assert len(self.lower) == len(self.upper)

        # Logging variables for objective and gradient values. For logging the latest variables are stored
        # as class parameters and appended to the log when calling the `logging()` function, in order to avoid
        # appending multiple values within one optimization step.
        self._constraint_current = None

        # Sanity checks.
        assert self.num_constraints == len(self.lower) == len(self.upper)

    @abstractmethod
    def initialize(self, **module_kwargs):
        raise NotImplementedError

    ###########################################################################
    # Constraint Formulation ##################################################
    ###########################################################################
    @abstractmethod
    def constraint_bounds(self) -> Tuple[Union[np.ndarray, List[None]], Union[np.ndarray, List[None]]]:
        raise NotImplementedError

    def constraint(self, ego_trajectory: torch.Tensor, ado_ids: List[str] = None) -> np.ndarray:
        """Determine constraint value for passed ego trajectory by calling the internal `compute()` method.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        assert check_ego_trajectory(x=ego_trajectory, pos_and_vel_only=True)
        constraint = self._compute(ego_trajectory, ado_ids=ado_ids)
        return self._return_constraint(constraint.detach().numpy())

    def jacobian(self, ego_trajectory: torch.Tensor, grad_wrt: torch.Tensor = None, ado_ids: List[str] = None) -> np.ndarray:
        """Determine jacobian matrix for passed ego trajectory. Therefore determine the constraint values by
        calling the internal `compute()` method and en passant build a computation graph. Then using the pytorch
        autograd library compute the jacobian matrix through the previously built computation graph.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param grad_wrt: vector w.r.t. which the gradient should be determined.
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        assert check_ego_trajectory(x=ego_trajectory, pos_and_vel_only=True)
        assert grad_wrt.requires_grad

        constraints = self._compute(ego_trajectory, ado_ids=ado_ids)
        grad_size = int(grad_wrt.numel())
        jacobian = torch.zeros(constraints.numel() * grad_size)
        for i, x in enumerate(constraints):
            grad = torch.autograd.grad(x, grad_wrt, retain_graph=True)
            jacobian[i*grad_size:(i+1)*grad_size] = grad[0].flatten().detach()
        return jacobian.detach().numpy()

    @abstractmethod
    def _compute(self, ego_trajectory: torch.Tensor, ado_ids: List[str] = None) -> torch.Tensor:
        raise NotImplementedError

    def compute_violation(self) -> float:
        """Determine constraint violation, i.e. how much the actual state is inside the constraint active region.
        When the constraint is not active, then the violation is zero. The calculation is based on the last (cached)
        evaluation of the constraint function.
        """
        no_violation = np.zeros(self.num_constraints)
        violation_lower = self.lower - self._constraint_current if None not in self.lower else no_violation
        violation_upper = self._constraint_current - self.upper if None not in self.upper else no_violation
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
    def inf_current(self) -> float:
        """Current infeasibility value (amount of constraint violation)."""
        return self.compute_violation()

    @property
    def num_constraints(self) -> int:
        raise NotImplementedError  # should be absolute input-independent number for sanity checking
