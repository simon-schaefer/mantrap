import numpy as np
import torch

from mantrap.simulation.simulation import GraphBasedSimulation
from mantrap.solver.cgrad.modules.module import Module
from mantrap.utility.maths import Derivative2


class InteractionAccelerationModule(Module):
    def __init__(self, env: GraphBasedSimulation, **module_kwargs):
        super(InteractionAccelerationModule, self).__init__(**module_kwargs)
        self._env = env
        self._ado_states_wo_np = self._env.predict(self.T, ego_trajectory=None).detach().numpy()
        self._derivative_2 = Derivative2(horizon=self.T, dt=self._env.dt, num_axes=4)
        self._ado_states_wo_ddt_np = self._derivative_2.compute(self._ado_states_wo_np[:, :, :, 0:2])

    def objective(self, x2: torch.Tensor) -> float:
        ado_states_np = self._env.predict(self.T, ego_trajectory=x2).detach().numpy()
        ado_states_ddt_np = self._derivative_2.compute(ado_states_np[:, :, :, 0:2])
        obj_value = np.linalg.norm(ado_states_ddt_np - self._ado_states_wo_ddt_np, axis=3).sum()
        return self._return_objective(obj_value)

    def gradient(self, x2: torch.Tensor) -> np.ndarray:
        gradient = np.zeros(2 * self.T)

        # Predict the ado states for the next `self.T` time-steps (although the full state is predicted here, we
        # merely need the agent's future positions going further.
        ado_states_np = self._env.predict(self.T, ego_trajectory=x2).detach().numpy()

        # Compute partial gradient using simulation graph derivative, expressing the correlation between the movement
        # of an ado and the ego's trajectory.
        partial_grads_np = self._build_partial_gradients(x2, env=self._env)

        # Compute gradient using analytically derived formula.
        ado_states_ddt_np = self._derivative_2.compute(ado_states_np[:, :, :, 0:2])
        diff = ado_states_ddt_np - self._ado_states_wo_ddt_np[:, :, :, 0:2]
        norm = np.linalg.norm(diff, axis=3)
        norm[norm < 1e-6] = np.inf  # if norm is zero, i.e. equivalent ddts, then gradient in this direction is 0 too
        for k in range(1, self.T):
            for t in range(k - 1, self.T - 1):
                partials_x = partial_grads_np[k, t - 1, :, 0] - 2 * partial_grads_np[k, t, :, 0]
                partials_y = partial_grads_np[k, t - 1, :, 1] - 2 * partial_grads_np[k, t, :, 1]
                gradient[2 * k] += np.sum(1 / norm[:, :, t] * np.sum(diff[:, :, t, 0], axis=1) * partials_x)
                gradient[2 * k + 1] += np.sum(1 / norm[:, :, t] * np.sum(diff[:, :, t, 1], axis=1) * partials_y)

        return self._return_gradient(gradient)
