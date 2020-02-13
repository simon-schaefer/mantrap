import numpy as np
import torch

from mantrap.simulation.simulation import GraphBasedSimulation
from mantrap.solver.modules.module import Module
from mantrap.utility.maths import Derivative2


class InteractionAccelerationModule(Module):
    def __init__(self, env: GraphBasedSimulation, **module_kwargs):
        super(InteractionAccelerationModule, self).__init__(**module_kwargs)
        self._env = env
        ado_states_wo = self._env.predict(self.T, ego_trajectory=None)
        self._derivative_2 = Derivative2(horizon=self.T, dt=self._env.dt, num_axes=2)
        self._ado_accelerations_wo = self._derivative_2.compute(ado_states_wo[:, :, :, 0:2])

    def objective(self, x2: torch.Tensor) -> float:
        obj_value = self._objective(x2)
        return self._return_objective(float(obj_value.item()))

    def gradient(self, x2: torch.Tensor, grad_wrt: torch.Tensor = None) -> np.ndarray:
        grad_wrt = x2 if grad_wrt is None else grad_wrt
        if not grad_wrt.requires_grad:
            grad_wrt.requires_grad = True

        objective = self._objective(x2)
        gradient = torch.autograd.grad(objective, grad_wrt)[0].flatten().detach().numpy()
        return self._return_gradient(gradient)

    def _objective(self, x2: torch.Tensor) -> torch.Tensor:
        graphs = self._env.build_connected_graph(ego_positions=x2, ego_grad=False)

        objective = torch.zeros(1)
        for k in range(1, self.T - 1):
            for m in range(self._env.num_ado_ghosts):
                ado_acceleration = self._derivative_2.compute_single(
                    graphs[f"{self._env.ado_ghosts[m].gid}_{k - 1}_position"],
                    graphs[f"{self._env.ado_ghosts[m].gid}_{k}_position"],
                    graphs[f"{self._env.ado_ghosts[m].gid}_{k}_position"],
                )
                ado_acceleration_wo = self._ado_accelerations_wo[m, 0, k, :]
                objective += torch.norm(ado_acceleration - ado_acceleration_wo)

        return objective
