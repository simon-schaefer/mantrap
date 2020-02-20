import torch

from mantrap.simulation.graph_based import GraphBasedSimulation
from mantrap.solver.objectives.objective_module import ObjectiveModule
from mantrap.utility.maths import Derivative2


class InteractionAccelerationModule(ObjectiveModule):
    def __init__(self, env: GraphBasedSimulation, **module_kwargs):
        super(InteractionAccelerationModule, self).__init__(**module_kwargs)
        self._env = env
        ado_states_wo = self._env.predict(self.T, ego_trajectory=None)
        self._derivative_2 = Derivative2(horizon=self.T, dt=self._env.dt, num_axes=2)
        self._ado_accelerations_wo = self._derivative_2.compute(ado_states_wo[:, :, :, 0:2])

    def _compute(self, x2: torch.Tensor) -> torch.Tensor:
        graphs = self._env.build_connected_graph(graph_input=x2, ego_grad=False)

        objective = torch.zeros(1)
        for k in range(1, self.T - 1):
            for m in range(self._env.num_ado_ghosts):
                ado_acceleration = self._derivative_2.compute_single(
                    graphs[f"{self._env.ado_ghosts[m].id}_{k - 1}_position"],
                    graphs[f"{self._env.ado_ghosts[m].id}_{k}_position"],
                    graphs[f"{self._env.ado_ghosts[m].id}_{k}_position"],
                )
                m_ado, m_mode = self._env.ghost_to_ado_index(m)
                ado_acceleration_wo = self._ado_accelerations_wo[m_ado, m_mode, k, :]
                objective += torch.norm(ado_acceleration - ado_acceleration_wo) * self._env.ado_ghosts[m].weight

        return objective
