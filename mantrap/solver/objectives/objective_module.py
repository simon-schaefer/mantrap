from abc import ABC, abstractmethod
import logging
from typing import List

import numpy as np
import torch

from mantrap.utility.shaping import check_ego_trajectory


class ObjectiveModule(ABC):
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
    def objective(self, ego_trajectory: torch.Tensor, ado_ids: List[str] = None) -> float:
        """Determine objective value for passed ego trajectory by calling the internal `compute()` method.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        assert check_ego_trajectory(x=ego_trajectory, pos_and_vel_only=True, t_horizon=self.T + 1)

        obj_value = self._compute(ego_trajectory, ado_ids=ado_ids)
        return self._return_objective(float(obj_value.item()))

    def gradient(self, ego_trajectory: torch.Tensor, grad_wrt: torch.Tensor, ado_ids: List[str] = None) -> np.ndarray:
        """Determine gradient vector for passed ego trajectory. Therefore determine the objective value by
        calling the internal `compute()` method and en passant build a computation graph. Then using the pytorch
        autograd library compute the gradient vector through the previously built computation graph.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param grad_wrt: vector w.r.t. which the gradient should be determined.
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        assert check_ego_trajectory(x=ego_trajectory, pos_and_vel_only=True, t_horizon=self.T + 1)
        assert grad_wrt.requires_grad
        assert ego_trajectory.requires_grad  # otherwise objective cannot have gradient function

        # Compute the objective value and check whether a gradient between the value and the ego_trajectory input
        # (which has been assured to require a gradient) exists, if the module-conditions for that are met.
        objective = self._compute(ego_trajectory, ado_ids=ado_ids)
        if self._objective_gradient_condition():
            assert objective.requires_grad

        # In general the objective might not be affected by the `ego_trajectory`, then it does not have a gradient
        # function and the gradient is not defined. Then the objective gradient is assumed to be zero.
        if objective.requires_grad:
            gradient = torch.autograd.grad(objective, grad_wrt, retain_graph=True, allow_unused=False)[0]
            gradient = gradient.flatten().detach().numpy()
        else:
            gradient = np.zeros(grad_wrt.numel())
        return self._return_gradient(gradient)

    @abstractmethod
    def _compute(self, ego_trajectory: torch.Tensor, ado_ids: List[str] = None) -> torch.Tensor:
        """Determine objective value core method.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        raise NotImplementedError

    @abstractmethod
    def _objective_gradient_condition(self) -> bool:
        """Conditions for the existence of a gradient between the input of the objective value computation
        (which is the ego_trajectory) and the objective value itself. If returns True and the ego_trajectory
        itself requires a gradient, the objective value output has to require a gradient as well.
        """
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

    ###########################################################################
    # Objective Properties ####################################################
    ###########################################################################
    @property
    def obj_current(self) -> float:
        return self._obj_current

    @property
    def grad_current(self) -> float:
        return np.linalg.norm(self._grad_current)
