import torch

from mantrap.simulation.simulation import GraphBasedSimulation
from mantrap.solver.objectives.objective_module import ObjectiveModule


class InteractionPositionModule(ObjectiveModule):
    def __init__(self, env: GraphBasedSimulation, **module_kwargs):
        super(InteractionPositionModule, self).__init__(**module_kwargs)
        assert env.num_ado_ghosts > 0 and env.ego is not None

        self._env = env
        self._ado_positions_wo = self._env.predict_wo_ego(t_horizon=self.T + 1)[:, :, :, 0:2]

    def _compute(self, x4: torch.Tensor) -> torch.Tensor:
        graphs = self._env.build_connected_graph(ego_trajectory=x4, ego_grad=False, ado_grad=False)

        objective = torch.zeros(1)
        for k in range(self.T):
            for m in range(self._env.num_ado_ghosts):
                ado_position = graphs[f"{self._env.ado_ghosts[m].id}_{k}_position"]
                m_ado, m_mode = self._env.ghost_to_ado_index(m)
                ado_position_wo = self._ado_positions_wo[m_ado, m_mode, k, :]
                objective += torch.norm(ado_position - ado_position_wo) * self._env.ado_ghosts[m].weight

        return objective
