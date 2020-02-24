from abc import abstractmethod
from collections import deque
import logging
from typing import List

import numpy as np
import torch

from mantrap.utility.shaping import check_ego_trajectory


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

    def constraint(self, x4: torch.Tensor) -> np.ndarray:
        assert check_ego_trajectory(ego_trajectory=x4, pos_and_vel_only=True)
        constraint = self._compute(x4)
        return self._return_constraint(constraint.detach().numpy())

    def jacobian(self, x4: torch.Tensor, grad_wrt: torch.Tensor = None) -> np.ndarray:
        assert check_ego_trajectory(ego_trajectory=x4, pos_and_vel_only=True)
        assert grad_wrt.requires_grad

        grad_size = int(grad_wrt.numel())
        constraint = self._compute(x4)
        jacobian = torch.zeros(constraint.numel() * grad_size)
        for i, x in enumerate(constraint):
            grad = torch.autograd.grad(x, grad_wrt, retain_graph=True)
            jacobian[i*grad_size:(i+1)*grad_size] = grad[0].flatten()
        return jacobian.detach().numpy()

    @abstractmethod
    def _compute(self, x4: torch.Tensor) -> torch.Tensor:
        pass

    ###########################################################################
    # Utility #################################################################
    ###########################################################################

    def _return_constraint(self, constraint_value: np.ndarray) -> np.ndarray:
        self._constraint_current = constraint_value
        logging.debug(f"Module {self.__str__()} computed")
        return self._constraint_current

    def logging(self):
        self._log_constraint.append(np.linalg.norm(self._constraint_current))

    def clean_up(self):
        self._log_constraint = deque()

    @property
    def logs(self) -> List[float]:
        return list(self._log_constraint)
