from abc import abstractmethod
import logging

import numpy as np
import torch

from mantrap.utility.shaping import check_ego_trajectory


class ObjectiveModule:

    def __init__(self, horizon: int, weight: float = 1.0, **module_kwargs):
        self.weight = weight
        self.T = horizon

        # Logging variables for objective and gradient values. For logging the latest variables are stored
        # as class parameters and appended to the log when calling the `logging()` function, in order to avoid
        # appending multiple values within one optimization step.
        self._obj_current, self._grad_current = 0.0, np.zeros(2)

    ###########################################################################
    # Optimization Formulation ################################################
    ###########################################################################

    def objective(self, x5: torch.Tensor) -> float:
        assert check_ego_trajectory(ego_trajectory=x5, pos_and_vel_only=True, t_horizon=self.T + 1)
        obj_value = self._compute(x5)
        return self._return_objective(float(obj_value.item()))

    def gradient(self, x5: torch.Tensor, grad_wrt: torch.Tensor) -> np.ndarray:
        assert check_ego_trajectory(ego_trajectory=x5, pos_and_vel_only=True, t_horizon=self.T + 1)
        assert grad_wrt.requires_grad

        objective = self._compute(x5)
        gradient = torch.autograd.grad(objective, grad_wrt, retain_graph=True)[0].flatten().detach().numpy()
        return self._return_gradient(gradient)

    @abstractmethod
    def _compute(self, x4: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    ###########################################################################
    # Utility #################################################################
    ###########################################################################
    def _return_objective(self, obj_value: float) -> float:
        self._obj_current = self.weight * obj_value
        logging.debug(f"Module {self.__str__()} with objective value {self._obj_current}")
        return self._obj_current

    def _return_gradient(self, gradient: np.ndarray) -> np.ndarray:
        logging.debug(f"Module {self.__str__()} with gradient {gradient}")
        self._grad_current = self.weight * gradient
        return self._grad_current

    @property
    def obj_current(self) -> float:
        return self._obj_current

    @property
    def grad_current(self) -> float:
        return np.linalg.norm(self._grad_current)
