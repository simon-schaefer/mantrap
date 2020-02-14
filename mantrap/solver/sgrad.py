from typing import List, Tuple

import numpy as np
import torch

from mantrap.constants import agent_speed_max
from mantrap.solver.ipopt_solver import IPOPTSolver
from mantrap.utility.shaping import check_trajectory_primitives


class SGradSolver(IPOPTSolver):

    ###########################################################################
    # Optimization formulation - Objective ####################################
    ###########################################################################
    """The objective is to minimize the interaction between the ego and ado, which can be expressed as the
    the L2 difference between the position of every agent with respect to interaction with ego and without
    taking it into account.

    J(x_{ego}) = sum_{t = 0}^T sum_{m = 0}^M || x_m^t(x_{ego}^t) - x_m_{wo}^t
    
    Taking the position-difference only into account sometimes results in a solution that makes the ego break the ado, 
    so intentionally going in front of the ado to decrease the distance between  x_m_{wo} and x_m. This may cause 
    collision and therefore is not a desired behaviour. Instead the differences in acceleration magnitude may be used, 
    which in general can be seen as a measure for the “comfort change” for the ado by inserting the robot in the scene. 
    
    J(x_{ego}) = sum_{t = 0}^T sum_{m = 0}^M || frac{delta^2}{delta t^2} (x_m^t - x_m_{wo}^t) ||

    using the Central Difference Expression (with error of order dt^2) we can extract the acceleration from the 
    positions assuming smoothness: d^2/dt^2 x_i = \frac{ x_{i + 1} - 2 x_i + x_{i - 1}}{ dt^2 }.
    """
    @staticmethod
    def objective_defaults() -> List[Tuple[str, float]]:
        return [("goal", 1.0), ("interaction", 1.0)]

    ###########################################################################
    # Optimization formulation - Constraints ##################################
    ###########################################################################
    """An unconstrained optimisation would result in pushing the ego as far as possible from the agents,
    ensuring minimal interaction but also leading to infinite (or at least very large) control effort on the
    one side and not reaching the goal in a reasonable amount of time on the other side. Therefore the trajectory
    length of the ego trajectory is bounded. The most exact but also most computationally expensive way doing
    that is by constraining subsequent trajectory points

    || x_{ego}^t - x_{ego}^{t - 1} || < gamma * dt

    for some maximal velocity gamma and the (simulation) time-step dt. However this would introduce T - 1
    quadratic (at least convex) constraints. Therefore another option would be to set a bound by constraining the
    sum over the whole trajectory only:

    sum_{t = 1}^T || x_{ego}^t - x_{ego}^{t - 1} || < gamma * T
    """
    def constraint_bounds(self, x_init: torch.Tensor) -> Tuple[List, List, List, List]:
        assert check_trajectory_primitives(x_init, t_horizon=self.T), "invalid initial trajectory"
        x_start = x_init[0, :].detach().numpy()

        # External constraint bounds (inter-point distance and initial point equality).
        cl = [None] * (self.T - 1) + [x_start[0], x_start[1]]
        cu = [agent_speed_max * self._env.dt] * (self.T - 1) + [x_start[0], x_start[1]]

        # Optimization variable bounds.
        lb = (np.ones(2 * self.T) * self._env.axes[0][0]).tolist()
        ub = (np.ones(2 * self.T) * self._env.axes[0][1]).tolist()

        return lb, ub, cl, cu

    @staticmethod
    def constraints_modules() -> List[str]:
        return ["path_length", "initial_point"]

    ###########################################################################
    # Utility #################################################################
    ###########################################################################
    def x_to_ego_trajectory(self, x: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        assert self._env.num_ado_modes == 1, "currently only uni-modal agents are supported"
        x2 = torch.from_numpy(x).view(self.T, 2)
        assert check_trajectory_primitives(x2, t_horizon=self.T), f"x should be ego trajectory with length {self.T}"
        return x2 if not return_leaf else (x2, x2)
