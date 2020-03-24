from typing import List, Tuple, Union

import numpy as np
import torch

from mantrap.constants import constraint_min_distance
from mantrap.solver.constraints.constraint_module import ConstraintModule


class MinDistanceModule(ConstraintModule):
    """Constraint for minimal distance between the robot (ego) and any other agent (ado) at any point in time.

    For computing the minimal distance between the ego and every ado the scene is forward simulated given the
    planned ego trajectory, using the `build_connected_graph()` method. Then the distance between ego and every ado
    is computed for every time-step of the trajectory. For 0 < t < T_{planning}:

    .. math:: || pos(t) - pos^{ado}_{0:2}(t) || > D

    :param horizon: planning time horizon in number of time-steps (>= 1).
    :param env: environment object for forward simulation of scene.
    """
    def __init__(self, horizon: int, **module_kwargs):
        self._env = None
        super(MinDistanceModule, self).__init__(horizon, **module_kwargs)

    def initialize(self, **module_kwargs):
        assert all([key in module_kwargs.keys() for key in ["env"]])
        self._env = module_kwargs["env"]

    def constraint_bounds(self) -> Tuple[Union[np.ndarray, List[None]], Union[np.ndarray, List[None]]]:
        return np.ones(self.num_constraints) * constraint_min_distance, [None] * self.num_constraints

    def _compute(self, x5: torch.Tensor) -> torch.Tensor:
        horizon = x5.shape[0]

        graphs = self._env.build_connected_graph(ego_trajectory=x5, ego_grad=False, ado_grad=False)
        constraints = torch.zeros((self._env.num_ado_ghosts, horizon))
        for m in range(self._env.num_ado_ghosts):
            for k in range(horizon):
                ado_position = graphs[f"{self._env.ado_ghosts[m].id}_{k}_position"]
                ego_position = x5[k, 0:2]
                constraints[m, k] = torch.norm(ado_position - ego_position)
        return constraints.flatten()

    @property
    def num_constraints(self) -> int:
        return (self.T + 1) * self._env.num_ado_ghosts
