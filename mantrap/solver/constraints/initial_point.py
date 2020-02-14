import torch

from mantrap.solver.constraints.constraint_module import ConstraintModule


class InitialPointModule(ConstraintModule):

    def __init__(self, **module_kwargs):
        super(InitialPointModule, self).__init__(**module_kwargs)

    def _compute(self, x2: torch.Tensor) -> torch.Tensor:
        return x2[0, :]
