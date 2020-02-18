from typing import List, Tuple

import numpy as np
import torch

from mantrap.constants import agent_speed_max
from mantrap.simulation.simulation import GraphBasedSimulation
from mantrap.solver.ipopt_solver import IPOPTSolver
from mantrap.utility.maths import lagrange_interpolation


class IGradSolver(IPOPTSolver):

    def __init__(self, sim: GraphBasedSimulation, goal: torch.Tensor, **solver_params):
        super(IGradSolver, self).__init__(sim, goal, **solver_params)

        # Number of control points determines the number of points used for interpolation, next to the initial
        # and terminal (goal) point. This is equivalent to the number of optimization variables (x2 -> 2D).
        if "num_control_points" not in self._solver_params.keys():
            self._solver_params["num_control_points"] = 2

    ###########################################################################
    # Initialization ##########################################################
    ###########################################################################
    def x0_default(self) -> torch.Tensor:
        steps = self.num_control_points + 2
        x0 = torch.stack((torch.linspace(self.env.ego.position[0].item(), self.goal[0].item(), steps=steps),
                          torch.linspace(self.env.ego.position[1].item(), self.goal[1].item(), steps=steps)), dim=1)
        return x0[1:-1, :]  # skip start and end point

    ###########################################################################
    # Optimization formulation - Objective ####################################
    ###########################################################################
    @staticmethod
    def objective_defaults() -> List[Tuple[str, float]]:
        return [("interaction", 1.0)]

    ###########################################################################
    # Optimization formulation - Constraints ##################################
    ###########################################################################
    def constraint_bounds(self, x_init: torch.Tensor) -> Tuple[List, List, List, List]:
        # External constraint bounds (interpolated inter-point distance).
        cl = [None] * (self.T - 1)
        cu = [agent_speed_max * self._env.dt] * (self.T - 1)

        # Optimization variable bounds.
        lb = (np.ones(2 * self.num_control_points) * self._env.axes[0][0]).tolist()
        ub = (np.ones(2 * self.num_control_points) * self._env.axes[0][1]).tolist()
        return lb, ub, cl, cu

    @staticmethod
    def constraints_modules() -> List[str]:
        return ["path_length"]

    ###########################################################################
    # Utility #################################################################
    ###########################################################################

    def x_to_ego_trajectory(self, x: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        mid = torch.tensor(x.astype(np.float64), requires_grad=True).view(self.num_control_points, 2).double()
        start_point = self._env.ego.position.unsqueeze(0)
        end_point = self._goal.unsqueeze(0)
        control_points = torch.cat((start_point, mid, end_point))

        interpolated = lagrange_interpolation(control_points, num_samples=self.T, deg=self.num_control_points + 2)
        return interpolated if not return_leaf else (interpolated, mid)

    @property
    def num_control_points(self) -> int:
        return self._solver_params["num_control_points"]
