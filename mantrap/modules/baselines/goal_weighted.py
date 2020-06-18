import typing

import numpy as np
import torch

import mantrap.environment

from ..goal_norm import GoalNormModule


class GoalWeightedModule(GoalNormModule):
    """Objective based on goal distance of every point of planned robot trajectory.

    This module merely serves as a baseline comparison for the `GoalMeanModule`, by using a weighted
    sum instead of the mean for combining the point-wise objectives. Following the idea that it is
    more important for the last point of the trajectory to be close to the goal than the first one,
    the larger the trajectory index the larger the weight of the point-wise distance.
    """
    def __init__(self, goal: torch.Tensor, env: mantrap.environment.base.GraphBasedEnvironment, **unused):
        super(GoalWeightedModule, self).__init__(goal, env=env, optimize_speed=False, **unused)

    def _objective_core(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str], tag: str
                        ) -> typing.Union[torch.Tensor, None]:
        goal_distances = torch.sum((ego_trajectory[:, 0:2] - self._goal).pow(2), dim=1)
        weights = torch.linspace(0.2, 1.0, steps=goal_distances.numel()).detach()
        return torch.mean(goal_distances * weights)

    def compute_gradient_analytically(
        self, ego_trajectory: torch.Tensor, grad_wrt: torch.Tensor, ado_ids: typing.List[str], tag: str
    ) -> typing.Union[np.ndarray, None]:
        return None

    ###########################################################################
    # Objective Properties ####################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "goal_sum"
