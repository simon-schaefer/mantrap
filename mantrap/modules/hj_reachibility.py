import os
import typing

import numpy as np
import scipy.io
import torch

import mantrap.agents
import mantrap.environment
import mantrap.utility.io
import mantrap.utility.shaping

from .base import OptimizationModule


class HJReachabilityModule(OptimizationModule):
    """Soft constraint based on Hamilton-Jacobi Reachability.

    Use HJ (aka backward) reachability to constraint the ego agent to not become "unsafer" when moving, i.e.
    to chose some trajectory so that the backward reachability value function does not become larger. This is
    equivalent to a min-max-game between the robot and each pedestrian, in which the pedestrian tries to catch
    the ego robot and while the robot tries to avoid the pedestrian. When the value function is negative, it
    is provably safe which means no matter what the pedestrian does (regarding safety trying to "catch" the robot is
    the worst case) they cannot reach each other.

    .. math::min \\nabla V(x)^T f_{rel}(x, u_R, u_P) - \\sigma \\geq 0
    .. math::\\sigma \\geq 0

    with `f_{rel}(x, u_R, u_P)` being the system dynamics of the relative state. Since the robot is modelled
    as double integrator while all pedestrians are modelled as single integrators, the relative (coupled) systems
    state only depends on the relative position as well as the velocity of the robot, not the velocity of the
    pedestrian (which is a system input).

    .. math::\\vec{x}_{rel} = \\begin{bmatrix} x_r - x_p \\ y_r - y_p \\ vx_r \\ vy_r \\end{bmatrix}

    The coupled system dynamics then are given as:

    .. math::f_{rel} = \\begin{bmatrix} vx_r - ux_p \\ vy_r - uy_p \\ ux_r \\ uy_r \\end{bmatrix}

    Since `sigma` is a slack variable the according weight in the objective function should be comparably large.
    """
    def __init__(self, env: mantrap.environment.base.GraphBasedEnvironment, t_horizon: int,  weight: float = 10.0,
                 data_file: str = "2D.mat", **unused):
        super(HJReachabilityModule, self).__init__(env=env, t_horizon=t_horizon, weight=weight,
                                                   has_slack=True, slack_weight=weight)

        # Pre-computed value function only for environment in which robot is double and each other
        # agent is a single integrator. However here we assume that the ados can be modelled as
        # single integrators anyway (e.g. by assuming a way faster reaction time coupled with
        # larger maximum acceleration than the robot).
        assert type(env.ego) == mantrap.agents.DoubleIntegratorDTAgent
        # assert all([type(ado) == mantrap.agents.IntegratorDTAgent for ado in env.ados()])

        # Read pre-computed value and gradient description for 2D case.
        data_directory = mantrap.utility.io.build_os_path("external/reachability")
        mat = scipy.io.loadmat(os.path.join(data_directory, data_file), squeeze_me=True)

        value_function = mat["value_function_flat"]
        gradients = mat["gradient_flat"]

        # Check same environment parameters in pre-computation and internal environment.
        # grid_min = (min_x, min_y, min_vx_robot, min_vy_robot)
        x_axis, y_axis = env.axes
        v_max_robot = env.ego.speed_max
        self._grid_min = mat["grid_min"].tolist()
        assert self._grid_min[2] == self._grid_min[3] == -v_max_robot
        self._grid_max = mat["grid_max"].tolist()
        assert self._grid_max[2] == self._grid_max[3] == v_max_robot
        # since relative state is a position difference, it could occur that the robot is at one
        # edge of the environment and the pedestrian at the other, or vice versa, therefore 2 x (!)
        assert self._grid_max[0] - self._grid_min[0] == 2 * (x_axis[1] - x_axis[0])
        assert self._grid_max[1] - self._grid_min[1] == 2 * (y_axis[1] - y_axis[0])

        # Check for synchronized time intervals.
        assert np.all(np.allclose(np.diff(mat["tau"]), np.ones(mat["tau"].size - 1) * env.dt))

        # Check non-saved coupled system parameters. #todo
        assert mantrap.constants.AGENT_SPEED_MAX == 4.0
        assert mantrap.constants.ROBOT_ACC_MAX == 2.0
        assert mantrap.constants.ROBOT_SPEED_MAX == 2.0

        # Re-shape value function and gradient into usable shape.
        # value_function = (t_horizon, dx, dy, vx, vy)
        # gradient = (4, t_horizon, dx, dy, vx, vy)
        self._num_points_by_dimension = mat["N"].tolist()
        self._value_function = np.reshape(value_function, (-1, *self._num_points_by_dimension))
        assert self._value_function.shape[0] >= t_horizon
        self._gradients = np.stack([np.reshape(gradient, (-1, *self._num_points_by_dimension))
                                   for gradient in gradients])
        assert self._gradients.shape[0] == 4  # for dimensions of relative state (dx, dy, vx_robot, vy_robot)
        assert self._gradients.shape[1] >= t_horizon

    ###########################################################################
    # Objective ###############################################################
    ###########################################################################
    def _compute_objective(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str], tag: str
                           ) -> typing.Union[torch.Tensor, None]:
        """Determine objective value core method.

        Since the module imposes soft constraints on the backward reachability value gradient, the
        objective value is introduced by the sum of slack variables, which is represented by
        `sigma` in constraint equation above. However soft constraints are dealt with automatically
        in the parent optimisation module class, therefore simply return `None` (no additional
        objective function) here.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        :param tag: name of optimization call (name of the core).
        """
        return None

    ###########################################################################
    # Constraint ##############################################################
    ###########################################################################
    def _compute_constraint(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str], tag: str
                            ) -> typing.Union[torch.Tensor, None]:
        """Determine constraint value core method.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        :param tag: name of optimization call (name of the core).
        """
        ego_controls = self._env.ego.roll_trajectory(ego_trajectory, dt=self._env.dt)
        u_robot = ego_controls[0, :].detach().numpy()
        t_max = self.t_horizon * self._env.dt

        ego_state, ado_states = self._env.states()
        ego_state = ego_state.detach().numpy()
        ado_states = ado_states.detach().numpy()
        constraints = np.zeros(len(ado_ids))
        for i_ado, ado_id in enumerate(ado_ids):
            m_ado = self._env.index_ado_id(ado_id=ado_id)
            u_max_ado = self._env.ados()[m_ado].speed_max

            # Determine current (i.e. at t = t0) relative state, which basically is the difference
            # in positions and the velocity of the ego robot. Since the pedestrians are modelled
            # as single integrators, as described above, their velocity is not part of the current
            # state, but a system input (!).
            state_rel = np.array([ego_state[0] - ado_states[m_ado, 0],
                                  ego_state[1] - ado_states[m_ado, 1],
                                  ego_state[2],
                                  ego_state[3]
                                  ])

            # In order to determine the system dynamics at the current state and given the worst
            # action the pedestrian could take, determine this worst action first.
            # The relative (coupled) system dynamics are described as shown in the module description.
            end_point_robot_wo = 0.5 * u_robot * t_max ** 2 + ego_state[2:4] * t_max + state_rel[0:2]
            u_ped_worst = np.sign(end_point_robot_wo) * np.minimum(np.abs(end_point_robot_wo / t_max),
                                                                   u_max_ado * np.ones(2))
            f_rel_worst = np.concatenate((ego_state[2:4] - u_ped_worst, u_robot))

            # Determine the value function's gradient at this current (relative) state.
            coords = self._convert_state_to_value_coordinates(state_rel)
            gradient = self._gradients[:, self.t_horizon, coords[0], coords[1], coords[2], coords[3]]
            constraints[i_ado] = np.matmul(gradient.T, f_rel_worst)
            # state_rel_next = state_rel + self._env.dt * f_rel_worst
            # coords_next = self._convert_state_to_value_coordinates(state_rel_next)
            # value_next = self._value_function[self.t_horizon, coords_next[0], coords_next[1], coords_next[2], coords_next[3]]
            # constraints[i_ado] = value_next

        return torch.from_numpy(constraints)

    def _constraint_boundaries(self) -> typing.Tuple[typing.Union[float, None], typing.Union[float, None]]:
        return 0.0, 0.0  # slack variable => inequality to equality constraint
        # return 0.0, None  # slack variable => inequality to equality constraint

    def _num_constraints(self, ado_ids: typing.List[str]) -> int:
        return len(ado_ids)

    ###########################################################################
    # Jacobian ################################################################
    ###########################################################################
    def _compute_jacobian_analytically(
        self, ego_trajectory: torch.Tensor, grad_wrt: torch.Tensor, ado_ids: typing.List[str], tag: str
    ) -> typing.Union[np.ndarray, None]:
        """Enforce usage of IPOPT automatic jacobian approximation, instead of manual computation."""
        return None

    ###########################################################################
    # Utility #################################################################
    ###########################################################################
    def _gradient_condition(self) -> bool:
        """Condition for back-propagating through the objective/constraint in order to obtain the
        objective's gradient vector/jacobian (numerically). If returns True and the ego_trajectory
        itself requires a gradient, the objective/constraint value, stored from the last computation
        (`_current_`-variables) has to require a gradient as well.

        The constraint and objective evaluation basically are table-lookups which are in general
        not differentiable (although there are approaches to do so, like some kind of function
        fitting, but they are not used here).
        """
        return False

    def _convert_state_to_value_coordinates(self, state: np.ndarray) -> np.ndarray:
        """Convert relative (not ego state  !) to value grid coordinates, i.e. the indices of the value
        function and value gradient grid. """
        value_coordinates = np.zeros(4)
        for i in range(4):
            coordinate_range = self._grid_max[i] - self._grid_min[i]
            n = self._num_points_by_dimension[i]
            value_coordinates[i] = int((state[i] - self._grid_min[i]) / coordinate_range * n)

        assert np.all(np.greater_equal(value_coordinates, np.zeros(4)))
        assert np.all(np.less(value_coordinates, self._num_points_by_dimension))
        return value_coordinates.astype(int)

    ###########################################################################
    # Module Properties #######################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "hj_reachability"
