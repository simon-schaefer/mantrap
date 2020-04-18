from typing import List, Tuple, Union

import torch

from mantrap.constants import *
from mantrap.environment.environment import GraphBasedEnvironment
from mantrap.solver.constraints.constraint_module import ConstraintModule


class MinDistanceModule(ConstraintModule):
    """Constraint for minimal distance between the robot (ego) and any other agent (ado) at any point in time.

    For computing the minimal distance between the ego and every ado the scene is forward simulated given the
    planned ego trajectory, using the `build_connected_graph()` method. Then the distance between ego and every ado
    is computed for every time-step of the trajectory. For 0 < t < T_{planning}:

    .. math:: || pos(t) - pos^{ado}_{0:2}(t) || > D

    :param horizon: planning time horizon in number of time-steps (>= 1).
    :param env: environment object for forward environment of scene.
    """
    def __init__(self, horizon: int, **module_kwargs):
        self._env = None
        super(MinDistanceModule, self).__init__(horizon, **module_kwargs)

    def initialize(self, env: GraphBasedEnvironment, **unused):
        self._env = env

    def _compute(self, ego_trajectory: torch.Tensor, ado_ids: List[str] = None) -> torch.Tensor:
        """Determine constraint value core method.

        Since predicting trajectories in the future the time-steps of the resulting time-discrete trajectories
        can assumed to be synced. However computing the inter-agent distance at the discrete time-steps only
        might not be accurate, especially in case of high velocities and large simulation time-steps `dt`. In
        regards of the increasing computational effort by introducing inter-step computations, i.e. constraining
        the distance between multiple steps between each time-step using interpolation (especially with regards
        on computing the Jacobian), only the distances between the ego and every other agent are compared
        at every discrete time-step, not in between time-steps.

        Therefore first the trajectories of all agents is predicted, then by iterating over all modes and time-steps
        in the discrete planning horizon, the constraint values are computed by calculating the L2 norm between
        the ego's and the predicted agents positions at this specific point in time.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        ado_ids = ado_ids if ado_ids is not None else self._env.ado_ids
        num_constraints_per_step = len(ado_ids) * self._env.num_modes
        horizon = ego_trajectory.shape[0]

        graph = self._env.build_connected_graph(ego_trajectory=ego_trajectory, ego_grad=False)
        constraints = torch.zeros((num_constraints_per_step, horizon))
        for m_ado, ado_id in enumerate(ado_ids):
            ghosts = self._env.ghosts_by_ado_id(ado_id=ado_id)
            for m_ghost, ghost in enumerate(ghosts):
                for t in range(horizon):
                    m = m_ado * len(ghosts) + m_ghost
                    ado_position = graph[f"{ghost.id}_{t}_{GK_POSITION}"]
                    ego_position = ego_trajectory[t, 0:2]
                    constraints[m, t] = torch.norm(ado_position - ego_position)
        return constraints.flatten()

    def _constraints_gradient_condition(self) -> bool:
        """Conditions for the existence of a gradient between the input of the constraint value computation
        (which is the ego_trajectory) and the constraint values itself. If returns True and the ego_trajectory
        itself requires a gradient, the constraint output has to require a gradient as well.

        The gradient between constraint values only exists if the internal environment is differentiable
        with respect to the ego (trajectory), the computation is conditioned on-
        """
        return self._env.is_differentiable_wrt_ego

    @property
    def constraint_bounds(self) -> Tuple[Union[float, None], Union[float, None]]:
        """Lower and upper bounds for constraint values.

        While there is no upper value for the distance, the lower bound is a constant minimal safety distance
        which is defined in constants.
        """
        return CONSTRAINT_MIN_L2_DISTANCE, None

    @property
    def num_constraints(self) -> int:
        return (self.T + 1) * self._env.num_ghosts
