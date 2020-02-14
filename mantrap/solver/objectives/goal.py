import torch

from mantrap.solver.objectives.objective_module import ObjectiveModule


class GoalModule(ObjectiveModule):

    def __init__(self, goal: torch.Tensor, **module_kwargs):
        assert goal.size() == torch.Size([2]), "goal position should be 2D vector"

        super(GoalModule, self).__init__(**module_kwargs)
        self._goal = goal
        self._distribution = torch.linspace(0, 1, steps=self.T) ** 2
        self._distribution = self._distribution / torch.sum(self._distribution)  # normalization (!)

    def _compute(self, x2: torch.Tensor) -> torch.Tensor:
        goal_distances = torch.norm(x2 - self._goal, dim=1)
        return goal_distances.dot(self._distribution)
