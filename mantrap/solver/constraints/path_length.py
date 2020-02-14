import torch

from mantrap.solver.constraints.constraint_module import ConstraintModule


class PathLengthModule(ConstraintModule):

    def __init__(self, **module_kwargs):
        super(PathLengthModule, self).__init__(**module_kwargs)

    def _compute(self, x2: torch.Tensor) -> torch.Tensor:
        return torch.norm(x2[1:, :] - x2[:-1, :], dim=1)
