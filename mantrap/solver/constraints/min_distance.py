from typing import Tuple

import numpy as np
import torch

from mantrap.constants import constraint_min_distance
from mantrap.solver.constraints.constraint_module import ConstraintModule


class MinDistanceModule(ConstraintModule):

    def __init__(self, **module_kwargs):
        assert all([key in module_kwargs.keys() for key in ["env"]])
        self._env = module_kwargs["env"]
        super(MinDistanceModule, self).__init__(**module_kwargs)

    def constraint_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        num_constraints = self.T * self._env.num_ado_ghosts
        return np.ones(num_constraints) * constraint_min_distance, [None] * num_constraints

    def _compute(self, x4: torch.Tensor) -> torch.Tensor:
        horizon = x4.shape[0]

        graphs = self._env.build_connected_graph(ego_trajectory=x4, ego_grad=False, ado_grad=False)
        constraints = torch.zeros((self._env.num_ado_ghosts, horizon))
        for m in range(self._env.num_ado_ghosts):
            for k in range(horizon):
                ado_position = graphs[f"{self._env.ado_ghosts[m].id}_{k}_position"]
                ego_position = x4[k, 0:2]
                constraints[m, k] = torch.norm(ado_position - ego_position)
        return constraints.flatten()
