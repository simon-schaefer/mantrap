import torch

from mantrap.solver.constraints.constraint_module import ConstraintModule


class MaxSpeedModule(ConstraintModule):

    def __init__(self, **module_kwargs):
        super(MaxSpeedModule, self).__init__(**module_kwargs)

    def _compute(self, x4: torch.Tensor) -> torch.Tensor:
        return x4[:, 2:4].flatten()
