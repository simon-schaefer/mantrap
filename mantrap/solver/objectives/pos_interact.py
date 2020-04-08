import torch

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

    def _compute(self, x5: torch.Tensor) -> torch.Tensor:
        graphs = self._env.build_connected_graph(trajectory=x5, ego_grad=False, ado_grad=False)

        objective = torch.zeros(1)
        for k in range(self.T):
            for m in range(self._env.num_ghosts):
                ghost_id = self._env.ghosts[m].id
                m_ado, m_mode = self._env.index_ghost_id(ghost_id=ghost_id)
                ado_position = graphs[f"{ghost_id}_{k}_position"]
                ado_position_wo = self._ado_positions_wo[m_ado, m_mode, k, :]
                objective += torch.norm(ado_position - ado_position_wo) * self._env.ghosts[m].weight

        return objective
