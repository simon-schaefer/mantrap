from abc import abstractmethod
import logging

import numpy as np
import torch

from mantrap.utility.shaping import check_ego_trajectory


class ObjectiveModule:
    """General objective class.

    For an unified and general implementation of the objective function modules this superclass implements methods
    for computing and logging the objective value as well as the gradient vector simply based on a single method,
    the `_compute()` method, which has to be implemented in the child classes. `_compute()` returns the objective value
    given a planned ego trajectory, while building a torch computation graph, which is used later on to determine the
    gradient vector using the PyTorch autograd library. Each output (objective & gradient) are multiplied with it's
    importance weight.

    :param horizon: planning time horizon in number of time-steps (>= 1).
    :param weight: objective importance weight.
    """
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
        """Determine objective value for passed ego trajectory `x5` by calling the internal `compute()` method.

        :param x5: planned ego trajectory (t_horizon, 5).
        """
        assert check_ego_trajectory(ego_trajectory=x5, pos_and_vel_only=True, t_horizon=self.T + 1)
        obj_value = self._compute(x5)
        return self._return_objective(float(obj_value.item()))

    def gradient(self, x5: torch.Tensor, grad_wrt: torch.Tensor) -> np.ndarray:
        """Determine gradient vector for passed ego trajectory `x5`. Therefore determine the objective value by
        calling the internal `compute()` method and en passant build a computation graph. Then using the pytorch
        autograd library compute the gradient vector through the previously built computation graph.

        :param x5: planned ego trajectory (t_horizon, 5).
        :param grad_wrt: vector w.r.t. which the gradient should be determined.
        """
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
