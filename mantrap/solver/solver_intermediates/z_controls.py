from abc import ABC

import numpy as np
import torch

from mantrap.solver.solver import Solver
from mantrap.utility.primitives import square_primitives
from mantrap.utility.shaping import check_ego_controls, check_ego_trajectory


class ZControlIntermediate(Solver, ABC):

    ###########################################################################
    # Initialization ##########################################################
    ###########################################################################
    def z0s_default(self, just_one: bool = False) -> torch.Tensor:
        """Initialize with three primitives, going from the current ego position to the goal point, following
        square shapes. The middle one (index = 1) is a straight line, the other two have some curvature,
        one positive and the other one negative curvature.

        When just one initial trajectory should be returned, then return straight line trajectory.
        """
        ego_path_init = square_primitives(start=self.env.ego.position, end=self.goal, steps=self.planning_horizon + 1)

        ego_controls_init = torch.zeros((ego_path_init.shape[0], self.planning_horizon, 2))
        for i, ego_path in enumerate(ego_path_init):
            ego_trajectory_init = self.env.ego.expand_trajectory(path=ego_path, dt=self.env.dt)
            ego_controls_init[i] = self.env.ego.roll_trajectory(trajectory=ego_trajectory_init, dt=self.env.dt)

        return ego_controls_init if not just_one else ego_controls_init[1].reshape(self.planning_horizon, 2)

    ###########################################################################
    # Problem formulation - Formulation #######################################
    ###########################################################################
    def num_optimization_variables(self) -> int:
        return 2 * self.planning_horizon

    ###########################################################################
    # Utility #################################################################
    ###########################################################################
    def z_to_ego_trajectory(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        ego_controls = torch.from_numpy(z).view(self.planning_horizon, 2)
        ego_controls.requires_grad = True
        ego_trajectory = self.env.ego.unroll_trajectory(controls=ego_controls, dt=self.env.dt)
        assert check_ego_trajectory(ego_trajectory, t_horizon=self.planning_horizon + 1, pos_and_vel_only=True)
        return ego_trajectory if not return_leaf else (ego_trajectory, ego_controls)

    def z_to_ego_controls(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        ego_controls = torch.from_numpy(z).view(self.planning_horizon, 2)
        ego_controls.requires_grad = True
        assert check_ego_controls(ego_controls, t_horizon=self.planning_horizon)
        return ego_controls if not return_leaf else (ego_controls, ego_controls)

    def ego_trajectory_to_z(self, ego_trajectory: torch.Tensor) -> np.ndarray:
        assert check_ego_trajectory(ego_trajectory)
        controls = self.env.ego.roll_trajectory(ego_trajectory, dt=self.env.dt)
        return controls.flatten().detach().numpy()
