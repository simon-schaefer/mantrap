from typing import List, Tuple

import numpy as np
import torch

from mantrap.solver.ipopt_solver import IPOPTSolver
from mantrap.utility.primitives import square_primitives
from mantrap.utility.shaping import check_ego_trajectory


class SGradSolver(IPOPTSolver):

    ###########################################################################
    # Initialization ##########################################################
    ###########################################################################
    def z0s_default(self, just_one: bool = False) -> torch.Tensor:
        x20s = square_primitives(start=self.env.ego.position, end=self.goal, dt=self.env.dt, steps=self.T)

        u0s = torch.zeros((x20s.shape[0], self.T - 1, 2))
        for i, x20 in enumerate(x20s):
            x40 = self.env.ego.expand_trajectory(path=x20, dt=self.env.dt)
            u0s[i] = self.env.ego.roll_trajectory(trajectory=x40, dt=self.env.dt)

        return u0s if not just_one else u0s[1].reshape(-1, 2)

    ###########################################################################
    # Problem formulation - Formulation #######################################
    ###########################################################################
    def num_optimization_variables(self) -> int:
        return self.T - 1

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
    @staticmethod
    def constraints_modules() -> List[str]:
        return ["max_speed"]

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
