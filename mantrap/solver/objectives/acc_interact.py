from typing import List

import torch

from mantrap.constants import *
from mantrap.environment.environment import GraphBasedEnvironment
from mantrap.solver.objectives.objective_module import ObjectiveModule
from mantrap.utility.maths import Derivative2


class InteractionAccelerationModule(ObjectiveModule):
    """Loss based on accelerational interaction between robot and ados.

    As a proxy for interaction based on the acceleration of every ado is computed in a (fictional) scene without an
    ego (robot) and compared to the actual occurring accelerations in the scene. As for autonomous driving the
    acceleration can be expressed "moving comfort", since a change in acceleration, especially a sudden change like
    strong de-acceleration, decreases the comfort of the agent.

    Re-Predicting it every time-step would be more correct, however it would also require a lot more computational
    effort (horizon times as much to be exact). Therefore merely the behavior of the ado without ego is computed
    that would occur, if the ego is not there from the beginning.

    .. math:: objective = \sum_{T} \sum_{ghosts} || acc_{t,i} - acc_{t,i}^{wo} ||_2

    :param env: environment for predicting the behaviour without interaction.
    """
    def __init__(self, env: GraphBasedEnvironment, **module_kwargs):
        super(InteractionAccelerationModule, self).__init__(**module_kwargs)
        assert env.num_ghosts > 0 and env.ego is not None

        self._env = env
        ado_states_wo = self._env.predict_wo_ego(t_horizon=self.T + 1)
        self._derivative_2 = Derivative2(horizon=self.T + 1, dt=self._env.dt, num_axes=2)
        self._ado_accelerations_wo = self._derivative_2.compute(ado_states_wo[:, :, :, 0:2])

    def _compute(self, ego_trajectory: torch.Tensor, ado_ids: List[str] = None) -> torch.Tensor:
        ado_ids = ado_ids if ado_ids is not None else self._env.ado_ids

        graph = self._env.build_connected_graph(ego_trajectory=ego_trajectory, ego_grad=False)

        objective = torch.zeros(1)
        for ado_id in ado_ids:
            for ghost in self._env.ghosts_by_ado_id(ado_id=ado_id):
                for t in range(1, self.T - 1):
                    i_ado, i_mode = self._env.convert_ghost_id(ghost_id=ghost.id)
                    ado_acceleration = self._derivative_2.compute_single(
                        graph[f"{ghost.id}_{t - 1}_{GK_POSITION}"],
                        graph[f"{ghost.id}_{t}_{GK_POSITION}"],
                        graph[f"{ghost.id}_{t}_{GK_POSITION}"],
                    )
                    ado_acceleration_wo = self._ado_accelerations_wo[i_ado, i_mode, t, :]
                    objective += torch.norm(ado_acceleration - ado_acceleration_wo) * ghost.weight

        return objective
