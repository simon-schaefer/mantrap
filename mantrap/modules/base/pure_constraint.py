import abc
import typing

import numpy as np
import torch

import mantrap.environment

from .optimization_module import OptimizationModule


class PureConstraintModule(OptimizationModule, abc.ABC):
    """Pure constraint module class.

    For an unified and general implementation of objective and constraint function modules, this superclass
    implements methods for computing both, either analytically or numerically based on the PyTorch autograd
    package. Thereby all objective and constraint computations should be purely based on the robot's (ego)
    trajectory, as well as the possibility to perform further roll-outs in a given simulation environment.

    The `PureConstraintModule` implements the general optimization module as pure constraint module, i.e.
    for hard constraints without any inter-connection to the objective function.
    """
    def __init__(self, t_horizon: int = None, env: mantrap.environment.base.GraphBasedEnvironment = None):
        super(PureConstraintModule, self).__init__(t_horizon=t_horizon, weight=0.0, env=env, has_slack=False)

    ###########################################################################
    # Objective ###############################################################
    ###########################################################################
    def _compute_objective(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str], tag: str
                           ) -> typing.Union[torch.Tensor, None]:
        """Returning `None` as an objective automatically ends objective and gradient computation
        and returns default values (zero). """
        return None

    ###########################################################################
    # Gradient ################################################################
    ###########################################################################
    def _compute_gradient_analytically(
        self, ego_trajectory: torch.Tensor, grad_wrt: torch.Tensor, ado_ids: typing.List[str], tag: str
    ) -> typing.Union[np.ndarray, None]:
        return None
