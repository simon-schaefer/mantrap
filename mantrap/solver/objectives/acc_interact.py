from typing import List, Union

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

    def _compute(self, ego_trajectory: torch.Tensor, ado_ids: List[str] = None) -> Union[torch.Tensor, None]:
        """Determine objective value core method.

        To compute the objective value first predict the behaviour of all agents (and modes) in the scene in the
        planning horizon, conditioned on the ego trajectory. Then iterate over every ghost in the scene and
        find the deviation between the acceleration of a specific agent at the specific point in time conditioned
        on the ego trajectory and unconditioned. Multiply by the weights of the modes, in order to encounter for
        difference in importance between these modes.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        # Per default (i.e. if `ado_ids`) is None use all ado ids defined in the environment.
        ado_ids = ado_ids if ado_ids is not None else self._env.ado_ids
        # The objective can only work if any ado agents are taken into account, otherwise return None.
        if len(ado_ids) == 0:
            return None

        # If more than zero ado agents are taken into account, compute the objective as described.
        # It is important to take all agents into account during the environment forward prediction step
        # (`build_connected_graph()`) to not introduce possible behavioural changes into the forward prediction,
        # which occur due to a reduction of the agents in the scene.
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

    def _objective_gradient_condition(self) -> bool:
        """Conditions for the existence of a gradient between the input of the objective value computation
        (which is the ego_trajectory) and the objective value itself. If returns True and the ego_trajectory
        itself requires a gradient, the objective value output has to require a gradient as well.

        If the internal environment is itself differentiable with respect to the ego (trajectory) input, the
        resulting objective value must have a gradient as well.
        """
        return self._env.is_differentiable_wrt_ego
