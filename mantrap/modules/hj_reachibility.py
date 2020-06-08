import os
import typing

import numpy as np
import scipy.interpolate
import scipy.io
import torch

import mantrap.agents
import mantrap.environment
import mantrap.utility.io
import mantrap.utility.maths
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
                 data_file: str = mantrap.constants.CONSTRAINT_HJ_MAT_FILE,
                 interp_method: str = mantrap.constants.CONSTRAINT_HJ_INTERPOLATION_METHOD,
                 **unused):
        super(HJReachabilityModule, self).__init__(env=env, t_horizon=t_horizon, weight=weight,
                                                   has_slack=True, slack_weight=weight)

        # Pre-computed value function only for environment in which robot is double and each other
        # agent is a single integrator. However here we assume that the ados can be modelled as
        # single integrators anyway (e.g. by assuming a way faster reaction time coupled with
        # larger maximum acceleration than the robot).
        assert type(env.ego) == mantrap.agents.DoubleIntegratorDTAgent
        # assert all([type(ado) == mantrap.agents.IntegratorDTAgent for ado in env.ados()])

        # Read pre-computed value and gradient description for 2D case.
        value_function, gradients, grid_size_by_dim, tau, (grid_min, grid_max) = self.unpack_mat_file(
            mat_file_path=os.path.join(mantrap.utility.io.build_os_path("third_party/reachability"), data_file)
        )

        # Check same environment parameters in pre-computation and internal environment.
        # Since relative state is a position difference, it could occur that the robot is at one
        # edge of the environment and the pedestrian at the other, or vice versa, therefore 2 x (!).
        # The pre-computed value function grid should be at least so large to cover these corner cases.
        # grid_min = (min_x, min_y, min_vx_robot, min_vy_robot)
        # grid_max = (min_x, min_y, min_vx_robot, min_vy_robot)
        x_axis, y_axis = env.axes
        v_min, v_max = env.ego.speed_limits
        assert grid_max[0] - grid_min[0] >= 2 * (x_axis[1] - x_axis[0])
        assert grid_max[1] - grid_min[1] >= 2 * (y_axis[1] - y_axis[0])
        assert grid_min[2] == grid_min[3] <= v_min
        assert grid_max[2] == grid_max[3] >= v_max
        self._grid_min, self._grid_max = grid_min, grid_max

        # Get value function time-intervals in order to determine which value function grid we need
        # when we want to make a statement about the full time horizon (planning horizon) of the
        # constraints in its (!) time-steps, they not necessarily have to be synced.
        safety_horizon = 1  # self.t_horizon
        t_horizon_s = safety_horizon * self._env.dt  # constraints time-horizon in seconds
        assert tau[-1] >= t_horizon_s
        t_value = int(np.argmax(tau > t_horizon_s))  # arg(value_function @ t = t_horizon in seconds)

        # Due to the curse of dimensionality the value function (and its gradient) cannot be computed with
        # sufficiently small grid resolution. Even if the pre-computed tensors would be very large in size
        # and would therefore take a lot of time to load (next to the blocked space in memory).
        # Thus, we use (linear) interpolation by exploiting the regular grid structure of the value function
        # tensor, implemented in the `scipy.interpolate` library.
        grid = [np.linspace(grid_min[i], grid_max[i], num=grid_size_by_dim[i]) for i in range(4)]
        vt = value_function[t_value, :, :, :, :]
        gt = gradients[:, t_value, :, :, :, :]
        self._value_function = scipy.interpolate.RegularGridInterpolator(grid, vt, method=interp_method)
        self._gradients = [scipy.interpolate.RegularGridInterpolator(grid, gt[i], method=interp_method)
                           for i in range(4)]

        # For analytical jacobian and debugging - store variables for auto-grad.
        self._x_rel = {}  # ado_id -> x_rel @ constraint

    ###########################################################################
    # Value Function ##########################################################
    ###########################################################################
    def value_function(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, self._grid_min, self._grid_max)
        return self._value_function(x)

    def value_gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, self._grid_min, self._grid_max)
        return np.concatenate([self._gradients[i](x) for i in range(4)])

    ###########################################################################
    # Objective ###############################################################
    ###########################################################################
    def objective_core(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str], tag: str
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
    def constraint_core(self, ego_trajectory: torch.Tensor, ado_ids: typing.List[str], tag: str,
                        enable_auto_grad: bool = False,
                        ) -> typing.Union[torch.Tensor, None]:
        """Determine constraint value core method.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param ado_ids: ghost ids which should be taken into account for computation.
        :param tag: name of optimization call (name of the core).
        :param enable_auto_grad: enable auto-grad to allow automatic backpropagation but slowing down
                                 computation (for debugging & testing only).
        """
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory)
        ego_controls = self.env.ego.roll_trajectory(ego_trajectory, dt=self._env.dt)
        ego_state, ado_states = self.env.states()

        with torch.set_grad_enabled(mode=enable_auto_grad):
            constraints = torch.zeros(len(ado_ids))
            u_robot = ego_controls[0, :]
            dt = self.env.dt

            # To compute the value function of the next state we have to compute the "worst-case" disturbance
            # first, which is in HJ Reachability, the disturbance that minimizes the value function (in our case).
            # So we could obtain it by solving an optimization problem, which probably is an overkill, but instead
            # we compute the value function for a grid of possible pedestrian controls (under some assumptions this
            # should approximate the minimum of the value function w.r.t. the disturbance `u_ped`).
            u_ped_max = mantrap.constants.PED_SPEED_MAX
            u_ped_grid = torch.linspace(-u_ped_max, u_ped_max, steps=10)
            u_ped_x, u_ped_y = torch.meshgrid([u_ped_grid, u_ped_grid])
            u_ped = torch.stack((u_ped_x.flatten(), u_ped_y.flatten())).reshape(-1, 2)

            for i_ado, ado_id in enumerate(ado_ids):
                m_ado = self.env.index_ado_id(ado_id=ado_id)
                x_ped = ado_states[m_ado, :]
                x_rel_next = self.state_relative(ego_state, u_r=u_robot, x_ped=x_ped, u_ped=u_ped, dt=dt)
                values = self.value_function(x=x_rel_next.detach().numpy())

                # The constraint is the minimum value function over all this possible disturbances (i.e. actions
                # of the pedestrian), ergo what is the value function if the pedestrian takes the worst action
                # regarding its safety.
                min_value_index = np.argmin(values)
                constraints[i_ado] = values[min_value_index]

                # For analytical jacobian and debugging store relative state value.
                self._x_rel[f"{tag}/{ado_id}"] = x_rel_next[min_value_index, :]

        return constraints

    def constraint_limits(self) -> typing.Tuple[typing.Union[float, None], typing.Union[float, None]]:
        return 0.0, 0.0  # slack variable => inequality to equality constraint

    def _num_constraints(self, ado_ids: typing.List[str]) -> int:
        return len(ado_ids)

    ###########################################################################
    # Jacobian ################################################################
    ###########################################################################
    def compute_jacobian_analytically(
        self, ego_trajectory: torch.Tensor, grad_wrt: torch.Tensor, ado_ids: typing.List[str], tag: str
    ) -> typing.Union[np.ndarray, None]:
        """Compute Jacobian matrix analytically.

        While the Jacobian matrix of the constraint can be computed automatically using PyTorch's automatic
        differentiation package there might be an analytic solution, which is when known for sure more
        efficient to compute. Although it is against the convention to use torch representations whenever
        possible, this function returns numpy arrays, since the main jacobian() function has to return
        a numpy array. Hence, not computing based on numpy arrays would just introduce an un-necessary
        `.detach().numpy()`.

        In the following we assume that assume that we look for the jacobian with respect to the ego controls,
        but to keep this general we will test it at the beginning. However since this module is based on look-up
        table interpolated values, we cannot compute the full derivative using torch's autograd framework only,
        as in other modules, therefore if `grad_wrt` is not the controls, an error is raised (since otherwise
        autograd would be used and fail).

        Since we pre-computed both the value function and its gradient computing the jacobian is quite
        straight-forward. Using  the chain rule we get:

        .. math:: \\frac{dJ}{du} = \\frac{dJ}{dx_{rel}} \\frac{dx_{rel}}{du}

        with dJ/dx_rel being pre-computed we only have to determine dx_rel/du.

        :param ego_trajectory: planned ego trajectory (t_horizon, 5).
        :param grad_wrt: vector w.r.t. which the gradient should be determined.
        :param ado_ids: ghost ids which should be taken into account for computation.
        :param tag: name of optimization call (name of the core).
        """
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory)

        with torch.no_grad():

            # Compute controls from trajectory, if not equal to `grad_wrt` return None.
            ego_controls = self.env.ego.roll_trajectory(ego_trajectory, dt=self.env.dt)
            if not mantrap.utility.maths.tensors_close(ego_controls.detach(), grad_wrt):
                raise NotImplementedError

            # By evaluating the constraints with the current input states we ensure that the internal
            # variables (relative states) are up-to-date.
            self.constraint_core(ego_trajectory=ego_trajectory, ado_ids=ado_ids, tag=tag, enable_auto_grad=False)

            # Otherwise compute Jacobian using formula in method's description above. The partial derivative
            t_horizon, u_size = ego_controls.shape
            jacobian = np.zeros((len(ado_ids), t_horizon * u_size))

            # dx_rel/du simply are zeros, except of two entries:
            # x_rel = x_rel^0 + dt * f_rel(v_r, u_p, u_r)
            # therefore the derivative dx_rel/du_r is zero, except of in the part of f_rel which depends on the
            # robot controls (last two entries). However only the first control action of the robot is used, hence
            # all other gradient entries are zero.
            dx_rel_du = np.zeros((4, t_horizon, u_size))
            dx_rel_du[2, 0, 0] = self.env.dt
            dx_rel_du[3, 0, 1] = self.env.dt
            dx_rel_du = dx_rel_du.reshape(4, -1)
            for i_ado, ado_id in enumerate(ado_ids):
                # Compute pre-computed gradient at evaluated relative state (see constraint_core).
                value_gradient = self.value_gradient(self.x_relative[f"{tag}/{ado_id}"])

                # Combine both partial gradients into the jacobian.
                jacobian[i_ado, :] = np.matmul(value_gradient, dx_rel_du)

        return jacobian

    ###########################################################################
    # Utility #################################################################
    ###########################################################################
    def gradient_condition(self) -> bool:
        """Condition for back-propagating through the objective/constraint in order to obtain the
        objective's gradient vector/jacobian (numerically). If returns True and the ego_trajectory
        itself requires a gradient, the objective/constraint value, stored from the last computation
        (`_current_`-variables) has to require a gradient as well.

        The constraint and objective evaluation basically are table-lookups which are in general
        not differentiable (although there are approaches to do so, like some kind of function
        fitting, but they are not used here).
        """
        return False

    @staticmethod
    def state_relative(x_r: torch.Tensor, u_r: torch.Tensor, x_ped: torch.Tensor, u_ped: torch.Tensor, dt: float
                       ) -> torch.Tensor:
        """Determine the relative state and dynamics (see module description):

        .. math::\\vec{x}_{rel} = \\begin{bmatrix} x_r - x_p \\ y_r - y_p \\ vx_r \\ vy_r \\end{bmatrix}
        .. math::f_{rel} = \\begin{bmatrix} vx_r - ux_p \\ vy_r - uy_p \\ ux_r \\ uy_r \\end{bmatrix}

        Since the states of robot and pedestrian as well as the action of the robot are known, at the time
        of evaluating the constraint, only the pedestrians controls are unknown. However in HJ reachability
        we look for the controls that minimize the value function, by evaluating the value function for
        several assignments of `u_ped` and finding the arg-min. In this context this function assumes that
        all states as well as the robots control are "unique" while it can handle a whole bunch of
        different pedestrian controls and computes the relative state for each of them.

        :param x_r: robot's state (x, y, vx, vy)
        :param u_r: robot's control input (ux, uy).
        :param x_ped: pedestrian's state (px, py, vpx, vpy).
        :param u_ped: pedestrian control input  (upx, upy).
        :param dt: time-step [s] for applying dynamics on current state.
        :returns: next state for each pedestrian control input.
        """
        assert mantrap.utility.shaping.check_ego_state(x_r, enforce_temporal=False)
        assert mantrap.utility.shaping.check_ego_action(u_r)
        assert mantrap.utility.shaping.check_ego_state(x_ped, enforce_temporal=False)
        assert mantrap.utility.shaping.check_ego_controls(u_ped)

        n = u_ped.shape[0]

        # Compute the current relative state and repeat it to the number of pedestrian controls.
        x_rel = torch.tensor([x_r[0] - x_ped[0], x_r[1] - x_ped[1], x_r[2], x_r[3]]).reshape(1, -1)
        x_rel_n = torch.mm(torch.ones((n, 1)), x_rel)

        # Compute dynamics stacked for all pedestrian controls and derive next state.
        u_r_n = torch.mm(torch.ones((n, 1)), u_r.reshape(1, -1))
        f_rel_n = torch.cat((x_r[2:4] - u_ped, u_r_n), dim=1)
        return x_rel_n + dt * f_rel_n

    @staticmethod
    def unpack_mat_file(mat_file_path: str) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                            typing.Tuple[np.ndarray, np.ndarray]]:
        mat = scipy.io.loadmat(mat_file_path, squeeze_me=True)
        grid_min, grid_max = mat["grid_min"], mat["grid_max"]
        value_tau = np.sort(mat["tau"].flatten())
        grid_size_by_dim = mat["N"]

        # Re-shape value function into usable shape and check for consistency.
        # value_function = (t_horizon, dx, dy, vx, vy)
        value_function = np.moveaxis(mat["value_function"], -1, 0)
        assert all(value_function.shape[1:] == grid_size_by_dim)

        # Re-shape gradients into usable shape and check for consistency.
        # gradient = (4, t_horizon, dx, dy, vx, vy)
        gradients = np.moveaxis(np.stack(mat["gradients"]), -1, 1)
        assert gradients.shape[0] == 4  # for dimensions of relative state (dx, dy, vx_robot, vy_robot)
        assert gradients.shape[1] == value_function.shape[0]
        assert all(gradients.shape[2:] == grid_size_by_dim)

        # Check non-saved coupled system parameters (for all variables it should hold min = -max).
        hyper_params = mat["params"]
        assert mantrap.constants.PED_SPEED_MAX == hyper_params["v_max_ped"]
        assert mantrap.constants.ROBOT_ACC_MAX == hyper_params["a_max_robot"]
        assert mantrap.constants.ROBOT_SPEED_MAX <= hyper_params["v_max_robot"]  # state-dimension not constraint

        return value_function, gradients, grid_size_by_dim, value_tau, (grid_min, grid_max)

    ###########################################################################
    # Module Properties #######################################################
    ###########################################################################
    @property
    def x_relative(self) -> typing.Dict[str, np.ndarray]:
        return self._x_rel

    @property
    def name(self) -> str:
        return "hj_reachability"
