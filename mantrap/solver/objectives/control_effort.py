from typing import List, Union

import torch

from mantrap.environment.environment import GraphBasedEnvironment
from mantrap.solver.objectives.objective_module import ObjectiveModule


class ControlEffortModule(ObjectiveModule):
    """Loss based on ego control effort.

    A typical objective in control theory is to reach some goal using a minimal amount of control energy (effort).
    In the use case of navigating through a crowd of pedestrians this objective is not priority, sacrificing some
    control energy for the sake of less interfering with pedestrians seems to be a feasible approach. Therefore
    constraining the amount of control energy which the robot is able to use in every step might be sufficient,
    however for completeness (and comparison) this objective is implemented:

    .. math:: objective =  \\sum_{T} || u_t ||_2

    :param env: solver's environment environment for ego dynamics.
    """
    def __init__(self, env: GraphBasedEnvironment, **module_kwargs):
        super(ControlEffortModule, self).__init__(**module_kwargs)
        self.initialize_env(env=env)

    def _compute(self, ego_trajectory: torch.Tensor, ado_ids: List[str] = None) -> Union[torch.Tensor, None]:
        """Determine objective value core method.

        Convert the ego trajectory to control inputs (using the ego agent's inverse dynamics) and derive
        the sum over the control norms over all time steps.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        ego_controls = self._env.ego.roll_trajectory(trajectory=ego_trajectory, dt=self._env.dt)
        return torch.sum(torch.norm(ego_controls[:, 0:2], dim=1))

    def _objective_gradient_condition(self) -> bool:
        """Conditions for the existence of a gradient between the input of the objective value computation
        (which is the ego_trajectory) and the objective value itself. If returns True and the ego_trajectory
        itself requires a gradient, the objective value output has to require a gradient as well.

        If the internal environment is itself differentiable with respect to the ego (trajectory) input, the
        resulting objective value must have a gradient as well.
        """
        return True
