import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from mantrap.constants import *
from mantrap.solver.solver import Solver
from mantrap.utility.shaping import check_ego_controls, check_ego_trajectory
from mantrap.utility.primitives import square_primitives


class MonteCarloTreeSearch(Solver):

    def _optimize(
        self,
        z0: torch.Tensor,
        ado_ids: List[str],
        tag: str = TAG_DEFAULT,
        max_iter: int = MCTS_MAX_STEPS,
        max_cpu_time: float = MCTS_MAX_CPU_TIME,
        **solver_kwargs
    ) -> Tuple[torch.Tensor, float, Dict[str, torch.Tensor]]:
        """Optimization function for single core to find optimal z-vector.

        Given some initial value `z0` find the optimal allocation for z with respect to the internally defined
        objectives and constraints. This function is executed in every thread in parallel, for different initial
        values `z0`. To simplify optimization not all agents in the scene have to be taken into account during
        the optimization but only the ones with ids defined in `ado_ids`.

        MCTS (Monte-Carlo-Tree-Search) uses random sampling during the full allowed computation time and
        returns the trajectory with the best expected objective value.

        :param z0: initial value of optimization variables.
        :param tag: name of optimization call (name of the core).
        :param ado_ids: identifiers of ados that should be taken into account during optimization.
        :returns: z_opt (optimal values of optimization variable vector)
                  objective_opt (optimal objective value)
                  optimization_log (logging dictionary for this optimization = self.log)
        """
        # Find variable bounds for random sampling during search.
        lb, ub = self.optimization_variable_bounds()
        lb, ub = np.asarray(lb), np.asarray(ub)

        # Start stopping conditions (runtime or number of iterations).
        sampling_start_time = time.time()
        sampling_iteration = 0

        # First of all evaluate the default trajectory as "baseline" for further trajectories.
        z_best = z0.detach().numpy()
        obj_best, _ = self._evaluate(z=z_best, tag=tag, ado_ids=ado_ids)

        # Then start sampling (MCTS) loop for finding more optimal trajectories.
        while sampling_iteration < max_iter and (time.time() - sampling_start_time) < max_cpu_time:
            z_sample = np.random.uniform(lb, ub)
            objective, constraint_violation = self._evaluate(z=z_sample, tag=tag, ado_ids=ado_ids)

            if obj_best > objective and constraint_violation < SOLVER_CONSTRAINT_LIMIT:
                obj_best = objective
                z_best = z_sample
            sampling_iteration += 1

        # The best sample is re-evaluated for logging purposes, since the last iteration is always assumed to
        # be the best iteration (logging within objective and constraint function).
        self._evaluate(z=z_best, tag=tag, ado_ids=ado_ids)
        return self.z_to_ego_controls(z=z_best), obj_best, self.log

    def _evaluate(self, z: np.ndarray, ado_ids: List[str], tag: str) -> Tuple[float, float]:
        objective = self.objective(z, tag=tag, ado_ids=ado_ids)
        _, constraint_violation = self.constraints(z, ado_ids=ado_ids, return_violation=True, tag=tag)
        return objective, constraint_violation

    ###########################################################################
    # Initialization ##########################################################
    ###########################################################################
    def initialize(self, **solver_params):
        pass

    def z0s_default(self, just_one: bool = False) -> torch.Tensor:
        start_pos, end_pos = self.env.ego.position, self.goal
        ego_path_init = square_primitives(start_pos, end_pos, dt=self.env.dt, steps=self.planning_horizon + 1)

        ego_controls_init = torch.zeros((ego_path_init.shape[0], self.planning_horizon, 2))
        for i, ego_path in enumerate(ego_path_init):
            ego_trajectory_init = self.env.ego.expand_trajectory(path=ego_path, dt=self.env.dt)
            ego_controls_init[i] = self.env.ego.roll_trajectory(trajectory=ego_trajectory_init, dt=self.env.dt)

        return ego_controls_init if not just_one else ego_controls_init[1].reshape(self.planning_horizon, 2)

    ###########################################################################
    # Problem formulation - Formulation #######################################
    ###########################################################################
    def num_optimization_variables(self) -> int:
        return self.planning_horizon

    ###########################################################################
    # Problem formulation - Objective #########################################
    ###########################################################################
    @staticmethod
    def objective_defaults() -> List[Tuple[str, float]]:
        return [(OBJECTIVE_GOAL, 1.0), (OBJECTIVE_INTERACTION_POS, 1.0)]

    ###########################################################################
    # Problem formulation - Constraints #######################################
    ###########################################################################
    @staticmethod
    def constraints_defaults() -> List[str]:
        return [CONSTRAINT_MAX_SPEED, CONSTRAINT_NORM_DISTANCE]

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
        assert check_ego_controls(x=ego_controls, t_horizon=self.planning_horizon)
        return ego_controls if not return_leaf else (ego_controls, ego_controls)

    def ego_trajectory_to_z(self, ego_trajectory: torch.Tensor) -> np.ndarray:
        assert check_ego_trajectory(ego_trajectory)
        controls = self.env.ego.roll_trajectory(ego_trajectory, dt=self.env.dt)
        return controls.flatten().detach().numpy()

    ###########################################################################
    # Solver properties #######################################################
    ###########################################################################
    @property
    def solver_name(self) -> str:
        return "mcts"
