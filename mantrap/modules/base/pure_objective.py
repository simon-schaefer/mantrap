import abc
import typing

import numpy as np
import torch

from .optimization_module import OptimizationModule


class PureObjectiveModule(OptimizationModule, abc.ABC):
    """Pure constraint module class.

    For an unified and general implementation of objective and constraint function modules, this superclass
    implements methods for computing both, either analytically or numerically based on the PyTorch autograd
    package. Thereby all objective and constraint computations should be purely based on the robot's (ego)
    trajectory, as well as the possibility to perform further roll-outs in a given simulation environment.

    The `PureConstraintModule` implements the general optimization module as pure constraint module, i.e.
    for hard constraints without any inter-connection to the objective function.
    """
    def __init__(self, t_horizon: int = None, weight: float = 1.0):
        super(PureObjectiveModule, self).__init__(t_horizon=t_horizon, weight=weight)

    ###########################################################################
    # Constraint & Jacobian ###################################################
    ###########################################################################
    def _compute_constraint(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str], tag: str
                            ) -> typing.Union[torch.Tensor, None]:
        """Returning `None` as an constraint automatically ends constraint and jacobian computation
        and returns default values (empty numpy array). """
        return None

    def _compute_jacobian_analytically(
            self, ego_trajectory: torch.Tensor, grad_wrt: torch.Tensor, ado_ids: typing.List[str], tag: str
    ) -> typing.Union[np.ndarray, None]:
        return None

    def _constraint_boundaries(self) -> typing.Tuple[typing.Union[float, None], typing.Union[float, None]]:
        return None, None

    def num_constraints(self, ado_ids: typing.List[str] = None) -> int:
        return 0
