import time
from typing import List, Tuple

import numpy as np
import torch

from mantrap.constants import mcts_max_cpu_time, mcts_max_steps, solver_constraint_limit
from mantrap.solver.solver import Solver
from mantrap.utility.shaping import check_ego_trajectory
from mantrap.utility.primitives import square_primitives


class MonteCarloTreeSearch(Solver):

    def determine_ego_controls(
        self,
        max_iter: int = mcts_max_steps,
        max_cpu_time: float = mcts_max_cpu_time,
        **solver_kwargs
    ) -> torch.Tensor:
        lb, ub = self.optimization_variable_bounds()
        lb, ub = np.asarray(lb), np.asarray(ub)

        sampling_start_time = time.time()
        sampling_iteration = 0

        # First of all evaluate the default trajectory as "baseline" for further trajectories.
        z0 = self.z0s_default(just_one=True).detach().numpy()
        obj_best, inf_best, x4_best, u2_best = self._evaluate(z=z0)

        # Then start sampling (MCTS) loop for finding more optimal trajectories.
        while sampling_iteration < max_iter and (time.time() - sampling_start_time) < max_cpu_time:
            z_sample = np.random.uniform(lb, ub)
            objective, constraint_violation, x4, u2 = self._evaluate(z=z_sample)

            if obj_best > objective and constraint_violation < solver_constraint_limit:
                obj_best = objective
                inf_best = constraint_violation
                x4_best, u2_best = x4.detach().clone(), u2.detach().clone()

            self.logging(x4=x4, u2=u2)
            sampling_iteration += 1

        self.logging(x4=x4_best, u2=u2_best)  # last results are assumed to be the best ones
        self.intermediate_log(iter_count=sampling_iteration, obj_value=obj_best, inf_value=inf_best)
        self.log_and_clean_up(tag=str(self._iteration), last_only=True, vis_keys=[])
        return u2_best

    def _evaluate(self, z: np.ndarray) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
        objective = self.objective(z)
        _, constraint_violation = self.constraints(z, return_violation=True)
        x4 = self.z_to_ego_trajectory(z=z)
        u2 = self.z_to_ego_controls(z=z)
        return objective, constraint_violation, x4, u2

    ###########################################################################
    # Initialization ##########################################################
    ###########################################################################
    def initialize(self, **solver_params):
        pass

    def z0s_default(self, just_one: bool = False) -> torch.Tensor:
        x20s = square_primitives(start=self.env.ego.position, end=self.goal, dt=self.env.dt, steps=self.T + 1)

        u0s = torch.zeros((x20s.shape[0], self.T, 2))
        for i, x20 in enumerate(x20s):
            x40 = self.env.ego.expand_trajectory(path=x20, dt=self.env.dt)
            u0s[i] = self.env.ego.roll_trajectory(trajectory=x40, dt=self.env.dt)

        return u0s if not just_one else u0s[1].reshape(self.T, 2)

    ###########################################################################
    # Problem formulation - Formulation #######################################
    ###########################################################################
    def num_optimization_variables(self) -> int:
        return self.T

    ###########################################################################
    # Problem formulation - Objective #########################################
    ###########################################################################
    @staticmethod
    def objective_defaults() -> List[Tuple[str, float]]:
        return [("goal", 1.0), ("interaction", 1.0)]

    ###########################################################################
    # Problem formulation - Constraints #######################################
    ###########################################################################
    @staticmethod
    def constraints_defaults() -> List[str]:
        return ["max_speed", "min_distance"]

    ###########################################################################
    # Utility #################################################################
    ###########################################################################
    def z_to_ego_trajectory(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        u2 = torch.from_numpy(z).view(self.T, 2)
        u2.requires_grad = True
        x4 = self.env.ego.unroll_trajectory(controls=u2, dt=self.env.dt)[:, 0:4]
        assert check_ego_trajectory(x4, t_horizon=self.T + 1, pos_and_vel_only=True)
        return x4 if not return_leaf else (x4, u2)

    def z_to_ego_controls(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        u2 = torch.from_numpy(z).view(self.T, 2)
        u2.requires_grad = True
        return u2 if not return_leaf else (u2, u2)
