from abc import abstractmethod
from collections import deque
import logging
from typing import List, Tuple, Union

import numpy as np
import torch

from mantrap.utility.shaping import check_ego_trajectory


class ConstraintModule:

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
        self._log_constraint = deque()

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

    def constraint(self, x4: torch.Tensor) -> np.ndarray:
        assert check_ego_trajectory(ego_trajectory=x4, pos_and_vel_only=True)
        constraint = self._compute(x4)
        return self._return_constraint(constraint.detach().numpy())

    def jacobian(self, x4: torch.Tensor, grad_wrt: torch.Tensor = None) -> np.ndarray:
        assert check_ego_trajectory(ego_trajectory=x4, pos_and_vel_only=True)
        assert grad_wrt.requires_grad

        constraints = self._compute(x4)
        grad_size = int(grad_wrt.numel())
        jacobian = torch.zeros(constraints.numel() * grad_size)
        for i, x in enumerate(constraints):
            grad = torch.autograd.grad(x, grad_wrt, retain_graph=True)
            jacobian[i*grad_size:(i+1)*grad_size] = grad[0].flatten().detach()
        return jacobian.detach().numpy()

    @abstractmethod
    def _compute(self, x4: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_violation(self) -> float:
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

    def logging(self):
        self._log_constraint.append(self.compute_violation())

    def clean_up(self):
        self._log_constraint = deque()

    ###########################################################################
    # Constraint Properties ###################################################
    ###########################################################################
    @property
    def logs(self) -> List[float]:
        return list(self._log_constraint)

    @property
    def num_constraints(self) -> int:
        raise NotImplementedError  # should be absolute input-independent number for sanity checking
