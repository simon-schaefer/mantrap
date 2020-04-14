from typing import List

import torch

from mantrap.constants import *
from mantrap.environment.environment import GraphBasedEnvironment
from mantrap.solver.objectives.objective_module import ObjectiveModule


class InteractionPositionModule(ObjectiveModule):
    """Loss based on positional interaction between robot and ados.

    As a proxy for interaction based on the position of every ado is computed in a (fictional) scene without an
    ego (robot) and compared to the actual occurring positions in the scene, as in intuitive measure for the change
    the robot's presence introduces to the scene.

    Re-Predicting it every time-step would be more correct, however it would also require a lot more computational
    effort (horizon times as much to be exact). Therefore merely the behavior of the ado without ego is computed
    that would occur, if the ego is not there from the beginning.

    .. math:: objective = \sum_{T} \sum_{ghosts} || pos_{t,i} - pos_{t,i}^{wo} ||_2

    :param env: solver's environment environment for predicting the behaviour without interaction.
    """
    def __init__(self, env: GraphBasedEnvironment, **module_kwargs):
        super(InteractionPositionModule, self).__init__(**module_kwargs)
        assert env.num_ghosts > 0 and env.ego is not None

        self._env = env
        self._ado_positions_wo = self._env.predict_wo_ego(t_horizon=self.T + 1)[:, :, :, 0:2]

    def _compute(self, ego_trajectory: torch.Tensor, ado_ids: List[str] = None) -> torch.Tensor:
        ado_ids = ado_ids if ado_ids is not None else self._env.ado_ids

        graph = self._env.build_connected_graph(ego_trajectory=ego_trajectory, ego_grad=False)

        objective = torch.zeros(1)
        for ado_id in ado_ids:
            for ghost in self._env.ghosts_by_ado_id(ado_id=ado_id):
                for t in range(1, self.T - 1):
                    m_ado, m_mode = self._env.convert_ghost_id(ghost_id=ghost.id)
                    ado_position = graph[f"{ghost.id}_{t}_{GK_POSITION}"]
                    ado_position_wo = self._ado_positions_wo[m_ado, m_mode, t, :]
                    objective += torch.norm(ado_position - ado_position_wo) * ghost.weight

        return objective
