from typing import List, Tuple

import numpy as np
import torch

from mantrap.constants import *
from mantrap.solver.ipopt_solver import IPOPTSolver
from mantrap.utility.primitives import square_primitives
from mantrap.utility.shaping import check_ego_controls, check_ego_trajectory


class SGradSolver(IPOPTSolver):
    """Shooting NLP using IPOPT solver.

    .. math:: z = controls
    .. math:: J(z) = J_{goal} + J_{interaction}
    .. math:: C(z) = [C_{max-speed}, C_{min-distance}]
    """

    ###########################################################################
    # Initialization ##########################################################
    ###########################################################################
    def initialize(self, **solver_params):
        pass

    def z0s_default(self, just_one: bool = False) -> torch.Tensor:
        ego_path_init = square_primitives(start=self.env.ego.position, end=self.goal, dt=self.env.dt, steps=self.T + 1)

        ego_controls_init = torch.zeros((ego_path_init.shape[0], self.T, 2))
        for i, ego_path in enumerate(ego_path_init):
            ego_trajectory_init = self.env.ego.expand_trajectory(path=ego_path, dt=self.env.dt)
            ego_controls_init[i] = self.env.ego.roll_trajectory(trajectory=ego_trajectory_init, dt=self.env.dt)

        return ego_controls_init if not just_one else ego_controls_init[1].reshape(self.T, 2)

    ###########################################################################
    # Problem formulation - Formulation #######################################
    ###########################################################################
    def num_optimization_variables(self) -> int:
        return self.T

    ###########################################################################
    # Optimization formulation - Objective ####################################
    ###########################################################################
    @staticmethod
    def objective_defaults() -> List[Tuple[str, float]]:
        return [(OBJECTIVE_GOAL, 1.0), (OBJECTIVE_INTERACTION, 10.0)]

    ###########################################################################
    # Optimization formulation - Constraints ##################################
    ###########################################################################
    @staticmethod
    def constraints_defaults() -> List[str]:
        return [CONSTRAINT_MAX_SPEED, CONSTRAINT_MIN_DISTANCE]

    ###########################################################################
    # Utility #################################################################
    ###########################################################################
    def z_to_ego_trajectory(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        ego_controls = torch.from_numpy(z).view(self.T, 2)
        ego_controls.requires_grad = True
        ego_trajectory = self.env.ego.unroll_trajectory(controls=ego_controls, dt=self.env.dt)

        assert check_ego_trajectory(ego_trajectory, t_horizon=self.T + 1, pos_and_vel_only=True)
        return ego_trajectory if not return_leaf else (ego_trajectory, ego_controls)

    def z_to_ego_controls(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        ego_controls = torch.from_numpy(z).view(self.T, 2)
        ego_controls.requires_grad = True

        assert check_ego_controls(ego_controls, t_horizon=self.T)
        return ego_controls if not return_leaf else (ego_controls, ego_controls)

    ###########################################################################
    # Logging parameters ######################################################
    ###########################################################################
    @property
    def solver_name(self) -> str:
        return "sgrad"
