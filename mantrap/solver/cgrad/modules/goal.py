import numpy as np
import torch

from mantrap.solver.cgrad.modules.module import Module


class GoalModule(Module):

    def __init__(self, goal: torch.Tensor, **module_kwargs):
        assert goal.size() == torch.Size([2]), "goal position should be 2D vector"

        super(GoalModule, self).__init__(**module_kwargs)
        self._goal_np = goal.detach().numpy()
        self._distribution = np.linspace(0, 1, num=self.T) ** 2
        self._distribution = self._distribution / np.sum(self._distribution)  # normalization (!)

    def objective(self, x2: torch.Tensor) -> float:
        goal_distances = np.linalg.norm(x2 - self._goal_np, axis=1)
        obj_value = goal_distances.dot(self._distribution)
        return self._return_objective(obj_value)

    def gradient(self, x2: torch.Tensor) -> np.ndarray:
        gradient = np.zeros(2 * self.T)

        diff_goal = x2 - self._goal_np
        norm_goal = np.linalg.norm(diff_goal, axis=1)
        for k in range(self.T):
            if norm_goal[k] < 1e-3:
                continue
            gradient[2 * k] += 1 / norm_goal[k] * diff_goal[k, 0] * self._distribution[k]
            gradient[2 * k + 1] += 1 / norm_goal[k] * diff_goal[k, 1] * self._distribution[k]

        return self._return_gradient(gradient)
