import torch

from mantrap.solver.objectives import GoalModule


def test_goal_distribution():
    goal = torch.tensor([4.1, 8.9])
    x2 = torch.rand((10, 2))

    module = GoalModule(goal=goal, horizon=10, weight=1.0)
    module.importance_distribution = torch.zeros(10)
    module.importance_distribution[3] = 1.0

    objective = module.objective(x2)
    assert objective == torch.norm(x2[3, :] - goal)
