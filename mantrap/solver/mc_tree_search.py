from typing import List, Tuple

import numpy as np
import torch

from mantrap.solver.solver import Solver
from mantrap.utility.shaping import check_ego_trajectory


class MonteCarloTreeSearch(Solver):

    def determine_ego_controls(self, **solver_kwargs) -> torch.Tensor:
        lb, ub = self.optimization_variable_bounds()
        lb, ub = np.asarray(lb), np.asarray(ub)

        control_samples = np.random.uniform(lb, ub)

    ###########################################################################
    # Problem formulation - Formulation #######################################
    ###########################################################################
    def num_optimization_variables(self) -> int:
        return self.T - 1

    ###########################################################################
    # Problem formulation - Objective #########################################
    ###########################################################################
    @staticmethod
    def objective_defaults() -> List[Tuple[str, float]]:
        return [("goal", 1.0), ("interaction", 1.0)]

    ###########################################################################
    # Problem formulation - Constraints #######################################
    ###########################################################################
    @staticmethod
    def constraints_modules() -> List[str]:
        return ["max_speed", "min_distance"]

    ###########################################################################
    # Utility #################################################################
    ###########################################################################
    def z_to_ego_trajectory(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        u2 = torch.from_numpy(z).view(self.T - 1, 2)
        u2.requires_grad = True
        x4 = self.env.ego.unroll_trajectory(controls=u2, dt=self.env.dt)[:, 0:4]
        assert check_ego_trajectory(x4, t_horizon=self.T, pos_and_vel_only=True)
        return x4 if not return_leaf else (x4, u2)

    def z_to_ego_controls(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        u2 = torch.from_numpy(z).view(self.T - 1, 2)
        u2.requires_grad = True
        return u2 if not return_leaf else (u2, u2)
