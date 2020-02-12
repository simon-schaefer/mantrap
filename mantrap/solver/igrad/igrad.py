from typing import List, Tuple

import numpy as np
import torch

from mantrap.simulation.simulation import GraphBasedSimulation
from mantrap.solver.solver import IPOPTSolver
from mantrap.utility.maths import lagrange_interpolation


class IGradSolver(IPOPTSolver):

    def __init__(self, sim: GraphBasedSimulation, goal: torch.Tensor, module: str = None, **solver_params):
        module = "interaction" if module is None else module
        super(IGradSolver, self).__init__(sim, goal, modules=[(module, 1.0)], **solver_params)

    ###########################################################################
    # Optimization formulation - Objective ####################################
    ###########################################################################
    def x_to_ego_trajectory(self, x: np.ndarray) -> torch.Tensor:
        assert self._env.num_ado_modes == 1, "currently only uni-modal agents are supported"
        mid = torch.tensor(x, requires_grad=True).float()
        points = torch.stack((self._env.ego.position, mid, self._goal)).float()
        return lagrange_interpolation(control_points=points, num_samples=self.T)

    ###########################################################################
    # Optimization formulation - Constraints ##################################
    ###########################################################################
    """Since basic properties of the path such as initial (and terminal) point are enforced by the interpolation 
    method itself, they do not have to be constrained. Therefore the only constraints are the optimization variables
    bounds, to be in the defined environment space.
    """

    def constraints(self, x: np.ndarray) -> np.ndarray:
        return np.array([])

    def constraint_bounds(self, x_init: torch.Tensor) -> Tuple[List, List, List, List]:
        # Optimization variable bounds.
        lb = (np.ones(2) * self._env.axes[0][0]).tolist()
        ub = (np.ones(2) * self._env.axes[0][1]).tolist()
        return lb, ub, [], []

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.array([])
