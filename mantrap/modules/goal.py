import typing

import numpy as np
import torch

import mantrap.utility.shaping

from .base import PureObjectiveModule


class GoalModule(PureObjectiveModule):
    """Objective based on goal distance of every point of planned robot trajectory.

    Next to avoiding interaction with other agents the robot should reach the goal state in a finite amount of
    time. Therefore the distance of every trajectory point to the goal state is taken to account, which is
    minimized the faster the robot gets to the goal.

    .. math:: objective = \\sum_{T} || pos_t - goal ||_2

    However, it is more important for the last rather than the first trajectory points to be close to the goal.
    Using some strictly-increasing distribution to weight the importance of the distance at every point in time
    did not lead to the expect result, while complicating the optimization. When we want to trade-off the
    goal cost with other cost, simply adapting its weight is sufficient as well.

    Additionally a cost for the velocity at the goal state can be included in this objective, a cost for non-zero
    velocity to be exact. This cost is weighted continuously based on the distance to the goal, i.e. the closer
    the a large speed (= velocity L2 norm) occurs, the higher its cost.

    .. math:: objective = \\sum_{T} w_t(d_{goal}(t)) || v_t ||_2

    :param goal: goal state/position for robot agent (2).
    :param optimize_speed: include cost for zero velocity at goal state.
    """

    def __init__(self, goal: torch.Tensor, optimize_speed: bool = False, **module_kwargs):
        super(GoalModule, self).__init__(**module_kwargs)

        assert mantrap.utility.shaping.check_goal(goal)
        self._goal = goal
        self._optimize_speed = optimize_speed

    def _compute_objective(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str] = None
                           ) -> typing.Union[torch.Tensor, None]:
        """Determine objective value core method.

        To compute the goal-based objective simply take the L2 norm between all positions on the ego trajectory
        and the goal. To encounter the fact, that it is more important for the last position (last = position at
        the end of the planning horizon) to be close to the goal position than the first position, multiply with
        a strictly increasing importance distribution, cubic in this case.

        When the goal-velocity is included here, compute the speed as L2 norm per trajectory state. Then weight
        the speeds at every time by its distance to the goal, as explained in the description of this module.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        goal_distances = torch.norm(ego_trajectory[:, 0:2] - self._goal, dim=1)
        cost = torch.mean(goal_distances)

        if self._optimize_speed:
            speeds = torch.norm(ego_trajectory[:, 2:4], dim=1)
            cost += speeds.dot(torch.exp(- goal_distances * 0.5))

        return cost

    def _compute_gradient_analytically(
        self, ego_trajectory: torch.Tensor, grad_wrt: torch.Tensor, ado_ids: typing.List[str] = None
    ) -> typing.Union[np.ndarray, None]:
        """Compute objective gradient vector analytically.

        While the gradient vector of the objective can be computed automatically using PyTorch's automatic
        differentiation package there might be an analytic solution, which is when known for sure more
        efficient to compute. Although it is against the convention to use torch representations whenever
        possible, this function returns numpy arrays, since the main gradient() function has to return
        a numpy array. Hence, not computing based on numpy arrays would just introduce an un-necessary
        `.detach().numpy()`.

        When no analytical solution is defined (or too hard to determine) return None.

        .. math::\\grad J = \\frac{dJ}{dz} = \\frac{dJ}{dx} \\frac{dx}{du}

        However it turned out that in fact the efficiency gains of the analytical solution, in comparison
        to the "numerical" solution are in the order of magnitude of 0.1 ms (about 0.4 ms for a time horizon
        of 10 time-steps). Given that the gradient is rarely computed, in comparison to the objective or
        the constraints, it is really not the bottleneck. For the sake of generality there the "numerical"
        solution is preferred.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param grad_wrt: vector w.r.t. which the gradient should be determined.
        :param ado_ids: ghost ids which should be taken into account for computation.
        """
        # assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory)
        #
        # with torch.no_grad():
        #     # Compute controls from trajectory, if not equal to `grad_wrt` return None.
        #     ego_controls = self._env.ego.roll_trajectory(ego_trajectory, dt=self._env.dt)
        #     if not ego_controls.shape == grad_wrt.shape and torch.all(torch.isclose(ego_controls, grad_wrt)):
        #         return None
        #
        #     # Compute dx/du from the agent's dynamics.
        #     dx_du = self._env.ego.dx_du(ego_controls, dt=self._env.dt).detach().numpy()
        #
        #     # Compute dJ/dx which is simply the derivative of the L2-norm over all positions.
        #     distance = ego_trajectory[:, 0:2] - self._goal
        #     t_horizon, d_size = distance.shape
        #     x_size = self._env.ego.state_size
        #
        #     d = distance.flatten().detach().numpy()
        #     d_norm = torch.norm(distance, dim=1).flatten().detach().numpy()
        #     d_norm[d_norm == 0.0] = 1e-6  # remove nan values when computing (1 / distance_norm)
        #     dJ_dx = d / np.repeat(d_norm, 2)
        #     dJ_dx /= dJ_dx.size / d_size  # axis-wise normalisation
        #
        #     # The goal objective only depends on positions, neither on the velocity nor the temporal
        #     # part of the state. Therefore stretch dJ_dx so that these parts (dJ/dv, dJ/dt) are zero.
        #     H = np.zeros((dJ_dx.size, dx_du.shape[0]))
        #     H[[i for i in range(dJ_dx.size)],
        #       [i // d_size * x_size + i % d_size for i in range(dJ_dx.size)]] = 1
        #     dJ_dx = np.matmul(dJ_dx, H)
        return None

    def _gradient_condition(self) -> bool:
        """Condition for back-propagating through the objective/constraint in order to obtain the
        objective's gradient vector/jacobian (numerically). If returns True and the ego_trajectory
        itself requires a gradient, the objective/constraint value, stored from the last computation
        (`_current_`-variables) has to require a gradient as well.

        Since the objective value computation depends on the ego_trajectory (and the ego goal) only, this
        should always hold.
        """
        return True

    ###########################################################################
    # Objective Properties ####################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "goal"
