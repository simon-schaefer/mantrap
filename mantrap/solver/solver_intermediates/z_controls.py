from abc import ABC

import numpy as np
import torch

from mantrap.solver.solver import Solver
from mantrap.utility.shaping import check_ego_controls, check_ego_trajectory


class ZControlIntermediate(Solver, ABC):

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

    def ego_controls_to_z(self, ego_controls: torch.Tensor) -> np.ndarray:
        assert check_ego_controls(ego_controls)
        return ego_controls.flatten().detach().numpy()
