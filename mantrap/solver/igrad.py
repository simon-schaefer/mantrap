from typing import List, Tuple

import numpy as np
import torch

from mantrap.solver.ipopt_solver import IPOPTSolver
from mantrap.utility.maths import lagrange_interpolation
from mantrap.utility.primitives import square_primitives
from mantrap.utility.shaping import check_ego_controls, check_ego_trajectory


class IGradSolver(IPOPTSolver):

    ###########################################################################
    # Initialization ##########################################################
    ###########################################################################
    def initialize(self, **solver_params):
        # Number of control points determines the number of points used for interpolation, next to the initial
        # and terminal (goal) point. This is equivalent to the number of optimization variables (x2 -> 2D).
        if "num_control_points" not in solver_params.keys():
            self._solver_params["num_control_points"] = 2

    def z0s_default(self, just_one: bool = False) -> torch.Tensor:
        x20s = square_primitives(self.env.ego.position, self.goal, dt=self.env.dt, steps=self.num_control_points + 2)
        return x20s[:, 1:-1, :] if not just_one else x20s[1, 1:-1, :]

    ###########################################################################
    # Problem formulation - Formulation #######################################
    ###########################################################################
    def num_optimization_variables(self) -> int:
        return self.num_control_points

    ###########################################################################
    # Optimization formulation - Objective ####################################
    ###########################################################################
    @staticmethod
    def objective_defaults() -> List[Tuple[str, float]]:
        return [("interaction", 1.0)]

    ###########################################################################
    # Optimization formulation - Constraints ##################################
    ###########################################################################
    @staticmethod
    def constraints_defaults() -> List[str]:
        return ["max_speed"]

    ###########################################################################
    # Utility #################################################################
    ###########################################################################
    def z_to_ego_trajectory(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        mid = torch.tensor(z).view(self.num_control_points, 2).float()
        mid.requires_grad = True

        start_point = self._env.ego.position.unsqueeze(0)
        end_point = self._goal.unsqueeze(0)
        control_points = torch.cat((start_point, mid, end_point))
        path = lagrange_interpolation(control_points, num_samples=self.T + 1, deg=self.num_control_points + 2)

        x4 = self.env.ego.expand_trajectory(path, dt=self.env.dt)[:, 0:4]
        assert check_ego_trajectory(x4, t_horizon=self.T + 1, pos_and_vel_only=True)
        return x4 if not return_leaf else (x4, mid)

    def z_to_ego_controls(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        x4, mid = self.z_to_ego_trajectory(z, return_leaf=True)
        u2 = self.env.ego.roll_trajectory(trajectory=x4, dt=self.env.dt)
        assert check_ego_controls(u2, t_horizon=self.T)
        return u2 if not return_leaf else (u2, mid)

    @property
    def num_control_points(self) -> int:
        return self._solver_params["num_control_points"]
