from typing import List, Tuple, Union

import torch

from mantrap.constants import *
from mantrap.environment.environment import GraphBasedEnvironment
from mantrap.solver.constraints.constraint_module import ConstraintModule


class MinDistanceModule(ConstraintModule):
    """Constraint for minimal distance between the robot (ego) and any other agent (ado) over all time.

    For computing the minimal distance between the ego and every ado the scene is forward simulated given the
    planned ego trajectory, using the `build_connected_graph()` method. Then the minimal distance between ego
    and every ado is computed for every time-step of the trajectory. For 0 < t < T_{planning}:

    .. math:: min_t || pos(t) - pos^{ado}_{0:2}(t) || > D

    In comparison to the `norm_distance` constraint, this constraint only includes the minimal value over all
    time-steps and agents (w.r.t. the ego agent) during the planning horizon. Thus, the `min_distance` constraint
    is a necessary condition for the `norm_distance` to be satisfied. However, although the `min_distance`
    constraint definitely is computationally way more efficient due to the decrease of  number of constraints
    (leading to way less backward passes), it gives a lot less information about where the optimization should go.
    In fact it reduces the problem of minimal distance to the "selected" agent and time-step, while the optimized
    trajectory has no gradients pointing to a certain distance from other agents at other points in time, so that
    the number of convergence steps might be increased.

    :param horizon: planning time horizon in number of time-steps (>= 1).
    :param env: environment object for forward environment of scene.
    """
    def __init__(self, horizon: int, **module_kwargs):
        self._env = None
        super(MinDistanceModule, self).__init__(horizon, **module_kwargs)

    def initialize(self, env: GraphBasedEnvironment, **unused):
        self._env = env

    ###########################################################################
    # Constraint Formulation ##################################################
    ###########################################################################
    def _compute(self, ego_trajectory: torch.Tensor, ado_ids: List[str] = None) -> Union[torch.Tensor, None]:
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
        the ego's and the predicted agents positions at this specific point in time. Afterwards the minimum over
        all these values is extracted and returned.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        ado_ids = ado_ids if ado_ids is not None else self._env.ado_ids
        num_constraints_per_step = len(ado_ids) * self._env.num_modes
        horizon = ego_trajectory.shape[0]
        # This constraint is only defined with respect to other agents, so if no other agents are taken into
        # account then return None directly.
        if len(ado_ids) == 0:
            return None

        # Otherwise compute the constraint as described above. It is important to take all agents into account
        # during the environment forward prediction step (`build_connected_graph()`) to not introduce possible
        # behavioural changes into the forward prediction, which occur due to a reduction of the agents in the
        # scene.
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
        return torch.min(constraints.flatten())

    def _constraints_gradient_condition(self) -> bool:
        """Conditions for the existence of a gradient between the input of the constraint value computation
        (which is the ego_trajectory) and the constraint values itself. If returns True and the ego_trajectory
        itself requires a gradient, the constraint output has to require a gradient as well.

        The gradient between constraint values only exists if the internal environment is differentiable
        with respect to the ego (trajectory), the computation is conditioned on-
        """
        return self._env.is_differentiable_wrt_ego

    ###########################################################################
    # Constraint Bounds #######################################################
    ###########################################################################
    @property
    def constraint_bounds(self) -> Tuple[Union[float, None], Union[float, None]]:
        """Lower and upper bounds for constraint values.

        While there is no upper value for the distance, the lower bound is a constant minimal
        safety distance which is defined in constants.
        """
        return CONSTRAINT_MIN_L2_DISTANCE, None

    def num_constraints(self, ado_ids: List[str] = None) -> int:
        return 1
