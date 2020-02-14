from abc import abstractmethod
from collections import deque
import logging
from typing import List

import numpy as np
import torch


class ConstraintModule:

    def __init__(self, **module_kwargs):
        # Logging variables for objective and gradient values. For logging the latest variables are stored
        # as class parameters and appended to the log when calling the `logging()` function, in order to avoid
        # appending multiple values within one optimization step.
        self._constraint_current = None
        self._log_constraint = deque()

    ###########################################################################
    # Constraint Formulation ##################################################
    ###########################################################################

    def constraint(self, x2: torch.Tensor) -> np.ndarray:
        constraint = self._compute(x2)
        return self._return_constraint(constraint.detach().numpy())

    def jacobian(self, x2: torch.Tensor, grad_wrt: torch.Tensor = None) -> np.ndarray:
        grad_wrt = x2 if grad_wrt is None else grad_wrt
        if not grad_wrt.requires_grad:
            grad_wrt.requires_grad = True

        grad_size = int(grad_wrt.numel())
        constraint = self._compute(x2)
        jacobian = torch.zeros(constraint.numel() * grad_size)
        for i, x in enumerate(constraint):
            grad = torch.autograd.grad(x, grad_wrt, retain_graph=True)
            jacobian[i*grad_size:(i+1)*grad_size] = grad[0].flatten()
        return jacobian.detach().numpy()

    @abstractmethod
    def _compute(self, x2: torch.Tensor) -> torch.Tensor:
        pass

    ###########################################################################
    # Utility #################################################################
    ###########################################################################

    def _return_constraint(self, constraint_value: np.ndarray) -> np.ndarray:
        self._constraint_current = constraint_value
        logging.debug(f"Module {self.__str__()} with constraint value {self._constraint_current}")
        return self._constraint_current

    def logging(self):
        self._log_constraint.append(np.linalg.norm(self._constraint_current))

    def clean_up(self):
        self._log_constraint = deque()

    @property
    def logs(self) -> List[float]:
        return list(self._log_constraint)
