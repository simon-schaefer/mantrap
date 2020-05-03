from typing import List, Union

import torch

from mantrap.solver.objectives.objective_module import ObjectiveModule
from mantrap.utility.shaping import check_goal


class GoalModule(ObjectiveModule):
    """Loss based on goal distance of every point of planned robot trajectory.

    Next to avoiding interaction with other agents the robot should reach the goal state in a finite amount of
    time. Therefore the distance of every trajectory point to the goal state is taken to account, which is
    minimized the faster the robot gets to the goal. However, it is more important for the last rather than the
    first trajectory points to be close to the goal, therefore the distance are weighted using a cubic distribution.

    .. math:: objective = \\sum_{T} w_t(t) || pos_t - goal ||_2

    Additionally a cost for the velocity at the goal state can be included in this objective, a cost for non-zero
    velocity to be exact. This cost is weighted continuously based on the distance to the goal, i.e. the closer
    the a large speed (= velocity L2 norm) occurs, the higher its cost.

    .. math:: objective = \\sum_{T} w_t(d_{goal}(t)) || v_t ||_2

    :param goal: goal state/position for robot agent (2).
    """
    def __init__(self, goal: torch.Tensor, **module_kwargs):
        assert check_goal(goal)

        super(GoalModule, self).__init__(**module_kwargs)
        self._goal = goal
        self._distribution = torch.linspace(0, 1, steps=self.t_horizon + 1) ** 3
        self._distribution = self._distribution / torch.sum(self._distribution)  # normalization (!)

    def _compute(self, ego_trajectory: torch.Tensor, ado_ids: List[str] = None) -> Union[torch.Tensor, None]:
        """Determine objective value core method.

        To compute the goal-based objective simply take the L2 norm between all positions on the ego trajectory
        and the goal. To encounter the fact, that it is more important for the last position (last = position at
        the end of the planning horizon) to be close to the goal position than the first position, multiply with
        a strictly increasing importance distribution, cubic in this case.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        goal_distances = torch.norm(ego_trajectory[:, 0:2] - self._goal, dim=1)
        return goal_distances.dot(self._distribution)

    def _objective_gradient_condition(self) -> bool:
        """Conditions for the existence of a gradient between the input of the objective value computation
        (which is the ego_trajectory) and the objective value itself. If returns True and the ego_trajectory
        itself requires a gradient, the objective value output has to require a gradient as well.

        Since the objective value computation depends on the ego_trajectory (and the ego goal) only, this
        should always hold.
        """
        return True

    @property
    def importance_distribution(self) -> torch.Tensor:
        return self._distribution

    @importance_distribution.setter
    def importance_distribution(self, weights: torch.Tensor):
        assert weights.numel() == self.t_horizon + 1
        self._distribution = weights
