from typing import List, Tuple

import numpy as np
import torch

from mantrap.constants import *
from mantrap.solver.solver_intermediates.ipopt import IPOPTIntermediate
from mantrap.utility.maths import lagrange_interpolation
from mantrap.utility.primitives import square_primitives
from mantrap.utility.shaping import check_ego_controls, check_ego_trajectory


class IGradIntermediate(IPOPTIntermediate):
    """Collocation NLP using IPOPT solver.

    .. math:: z = lagrange-parameters
    .. math:: J(z) = J_{interaction}
    .. math:: C(z) = [C_{max-speed}]
    """

    ###########################################################################
    # Initialization ##########################################################
    ###########################################################################
    def initialize(self, **solver_params):
        # Number of control points determines the number of points used for interpolation, next to the initial
        # and terminal (goal) point. This is equivalent to the number of optimization variables (x2 -> 2D).
        if PK_NUM_CONTROL_POINTS not in solver_params.keys():
            self._solver_params[PK_NUM_CONTROL_POINTS] = 2

    def z0s_default(self, just_one: bool = False) -> torch.Tensor:
        ego_path_init = square_primitives(start=self.env.ego.position, end=self.goal, steps=self.num_control_points + 2)
        return ego_path_init[:, 1:-1, :] if not just_one else ego_path_init[1, 1:-1, :]

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
        return [(OBJECTIVE_INTERACTION_POS, 1.0)]

    ###########################################################################
    # Optimization formulation - Constraints ##################################
    ###########################################################################
    @staticmethod
    def constraints_defaults() -> List[str]:
        return [CONSTRAINT_MAX_SPEED]

    ###########################################################################
    # Utility #################################################################
    ###########################################################################
    def z_to_ego_trajectory(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        mid = torch.tensor(z).view(self.num_control_points, 2).float()
        mid.requires_grad = True

        start_point = self._env.ego.position.unsqueeze(0)
        end_point = self._goal.unsqueeze(0)
        control_points = torch.cat((start_point, mid, end_point))
        path = lagrange_interpolation(
            control_points,
            num_samples=self.planning_horizon + 1,
            deg=self.num_control_points + 2
        )
        ego_trajectory = self.env.ego.expand_trajectory(path, dt=self.env.dt)

        assert check_ego_trajectory(ego_trajectory, t_horizon=self.planning_horizon + 1, pos_and_vel_only=True)
        return ego_trajectory if not return_leaf else (ego_trajectory, mid)

    def z_to_ego_controls(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        ego_trajectory, mid = self.z_to_ego_trajectory(z, return_leaf=True)
        ego_controls = self.env.ego.roll_trajectory(trajectory=ego_trajectory, dt=self.env.dt)
        assert check_ego_controls(ego_controls, t_horizon=self.planning_horizon)
        return ego_controls if not return_leaf else (ego_controls, mid)

    def ego_trajectory_to_z(self, ego_trajectory: torch.Tensor) -> np.ndarray:
        assert check_ego_trajectory(ego_trajectory)

        # An inverse Lagrange interpolation is not invertible. However Lagrange interpolation guarantees that
        # the resulting trajectory intersects with all its control points, so the points are somewhere on the
        # trajectory, but we don't know where exactly. The best guess we have is equally spacing the points
        # on the trajectory (index-wise), while the first and last points always are part of the trajectory.
        indexes = np.linspace(start=0, endpoint=self.planning_horizon, num=self.num_control_points + 2)
        return ego_trajectory[indexes, 0:2].flatten().detach().numpy()

    @property
    def num_control_points(self) -> int:
        return self._solver_params[PK_NUM_CONTROL_POINTS]

    ###########################################################################
    # Solver properties #######################################################
    ###########################################################################
    @property
    def solver_name(self) -> str:
        return "igrad"
