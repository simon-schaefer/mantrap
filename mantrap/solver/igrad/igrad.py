import logging
from typing import List, Tuple

import numpy as np
import torch

from mantrap.simulation.simulation import GraphBasedSimulation
from mantrap.solver.solver import IPOPTSolver
from mantrap.utility.maths import lagrange_interpolation


class IGradSolver(IPOPTSolver):

    def __init__(self, sim: GraphBasedSimulation, goal: torch.Tensor, **solver_params):
        super(IGradSolver, self).__init__(sim, goal, **solver_params)

        # For objective function evaluation store the ado trajectories without interaction.
        self._ado_states_wo = self._env.predict(self.T, ego_trajectory=None)

    ###########################################################################
    # Optimization formulation - Objective ####################################
    ###########################################################################
    def objective(self, x: np.ndarray) -> float:
        x2 = self.x_to_ego_trajectory(x)
        graphs = self._env.build_connected_graph(ego_positions=x2, ego_grad=False)

        objective = torch.zeros(1)
        for k in range(self.T):
            for m in range(self._env.num_ado_ghosts):
                ado_position = graphs[f"{self._env.ado_ghosts[m].gid}_{k}_position"]
                ado_position_wo = self._ado_states_wo[m, 0, k, 0:2]
                objective += torch.norm(ado_position - ado_position_wo)

        logging.debug(f"Objective function = {objective}")
        if self.is_verbose:
            self._x_latest = x2.detach().numpy().copy()  # logging most current optimization values
        return float(objective.item())

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x2, x_tensor = self.x_to_ego_trajectory(x, return_x_tensor=True)
        graphs = self._env.build_connected_graph(ego_positions=x2, ego_grad=False)

        objective = torch.zeros(1)
        for k in range(self.T):
            for m in range(self._env.num_ado_ghosts):
                ado_position = graphs[f"{self._env.ado_ghosts[m].gid}_{k}_position"]
                ado_position_wo = self._ado_states_wo[m, 0, k, 0:2]
                objective += torch.norm(ado_position - ado_position_wo)

        gradient = torch.autograd.grad(objective, x_tensor)[0].detach().numpy()

        logging.debug(f"Gradient function = {gradient}")
        if self.is_verbose:
            self._x_latest = x2.detach().numpy().copy()  # logging most current optimization values
        return gradient

    ###########################################################################
    # Optimization formulation - Constraints ##################################
    ###########################################################################
    """Since basic properties of the path such as initial (and terminal) point are enforced by the interpolation 
    method itself, they do not have to be constrained. Therefore the only constraints are the optimization variables
    bounds, to be in the defined environment space.
    """

    def constraints(self, x: np.ndarray) -> np.ndarray:
        return np.array([])

    def constraint_bounds(self, x_init: torch.Tensor) -> Tuple[List, List, List, List]:
        # Optimization variable bounds.
        lb = (np.ones(2) * self._env.axes[0][0]).tolist()
        ub = (np.ones(2) * self._env.axes[0][1]).tolist()
        return lb, ub, [], []

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.array([])

    ###########################################################################
    # Utility #################################################################
    ###########################################################################

    def x_to_ego_trajectory(self, x: np.ndarray, return_x_tensor: bool = False) -> torch.Tensor:
        assert self._env.num_ado_modes <= 1, "currently only uni-modal agents are supported"
        mid = torch.tensor(x.astype(np.float64), requires_grad=True).double()
        points = torch.stack((self._env.ego.position, mid, self._goal))
        interpolated = lagrange_interpolation(control_points=points, num_samples=self.T)
        return interpolated if not return_x_tensor else (interpolated, mid)
