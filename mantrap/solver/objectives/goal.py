import torch

from mantrap.solver.objectives.objective_module import ObjectiveModule


class GoalModule(ObjectiveModule):
    """Loss based on goal distance of every point of planned robot trajectory.

    Next to avoiding interaction with other agents the robot should reach the goal state in a finite amount of
    time. Therefore the distance of every trajectory point to the goal state is taken to account, which is
    minimized the faster the robot gets to the goal. However, it is more important for the last rather than the
    first trajectory points to be close to the goal, therefore the distance are weighted using a cubic distribution.

    .. math:: objective = \sum_{T} w_t || pos_t - goal ||_2

    :param goal: goal state/position for robot agent (2).
    """
    def __init__(self, goal: torch.Tensor, **module_kwargs):
        assert goal.size() == torch.Size([2])

        super(GoalModule, self).__init__(**module_kwargs)
        self._goal = goal
        self._distribution = torch.linspace(0, 1, steps=self.T + 1) ** 3
        self._distribution = self._distribution / torch.sum(self._distribution)  # normalization (!)

    def _compute(self, x5: torch.Tensor) -> torch.Tensor:
        goal_distances = torch.norm(x5[:, 0:2] - self._goal, dim=1)
        return goal_distances.dot(self._distribution)

    @property
    def importance_distribution(self) -> torch.Tensor:
        return self._distribution

    @importance_distribution.setter
    def importance_distribution(self, weights: torch.Tensor):
        assert weights.numel() == self.T + 1, "distribution has invalid length"
        self._distribution = weights
