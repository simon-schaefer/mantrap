from typing import List, Tuple

import numpy as np
import torch

from mantrap.constants import agent_speed_max
from mantrap.simulation.graph_based import GraphBasedSimulation
from mantrap.solver.ipopt_solver import IPOPTSolver
from mantrap.utility.maths import lagrange_interpolation
from mantrap.utility.primitives import straight_line


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
    def z0_default(self) -> torch.Tensor:
        x0 = straight_line(self.env.ego.position, self.goal, steps=self.num_control_points + 2)
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
    def constraint_bounds(self) -> Tuple[List, List, List, List]:
        # External constraint bounds (interpolated inter-point distance).
        cl = [None] * (self.T - 1)
        cu = [agent_speed_max * self._env.dt] * (self.T - 1)

        # Optimization variable bounds.
        lb = (np.ones(2 * self.num_control_points) * self._env.axes[0][0]).tolist()
        ub = (np.ones(2 * self.num_control_points) * self._env.axes[0][1]).tolist()
        return lb, ub, cl, cu

    @staticmethod
    def constraints_modules() -> List[str]:
        return ["max_speed"]

    ###########################################################################
    # Utility #################################################################
    ###########################################################################

    def z_to_ego_trajectory(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        mid = torch.tensor(z.astype(np.float64)).view(self.num_control_points, 2).double()
        mid.requires_grad = True

        start_point = self._env.ego.position.unsqueeze(0)
        end_point = self._goal.unsqueeze(0)
        control_points = torch.cat((start_point, mid, end_point))
        path = lagrange_interpolation(control_points, num_samples=self.T, deg=self.num_control_points + 2)

        x4 = self.env.ego.expand_trajectory(path, dt=self.env.dt, t_start=self.env.sim_time)[:, 0:4]
        return x4 if not return_leaf else (x4, mid)

    @property
    def num_control_points(self) -> int:
        return self._solver_params["num_control_points"]
