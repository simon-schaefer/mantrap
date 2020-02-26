from typing import List, Tuple

import numpy as np
import torch

from mantrap.solver.ipopt_solver import IPOPTSolver
from mantrap.utility.primitives import square_primitives
from mantrap.utility.shaping import check_ego_trajectory


class CGradSolver(IPOPTSolver):

    ###########################################################################
    # Initialization ##########################################################
    ###########################################################################
    def z0s_default(self) -> torch.Tensor:
        x20s = square_primitives(start=self.env.ego.position, end=self.goal, dt=self.env.dt, steps=self.T)
        u0s = torch.zeros((x20s.shape[0], self.T - 1, 2))
        for i, x20 in enumerate(x20s):
            x40 = self.env.ego.expand_trajectory(path=x20, dt=self.env.dt)
            u0s[i] = self.env.ego.roll_trajectory(trajectory=x40, dt=self.env.dt)
        return u0s

    ###########################################################################
    # Optimization formulation - Objective ####################################
    ###########################################################################
    @staticmethod
    def objective_defaults() -> List[Tuple[str, float]]:
        return [("goal", 1.0), ("interaction", 1.0)]

    ###########################################################################
    # Optimization formulation - Constraints ##################################
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
