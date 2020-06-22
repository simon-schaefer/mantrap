import abc
import logging
import typing

import numpy as np
import torch

import mantrap.constants
import mantrap.environment
import mantrap.attention
import mantrap.modules
import mantrap.utility

from .logging import OptimizationLogger


class TrajOptSolver(abc.ABC):
    """General abstract solver implementation.

    The idea of this general implementation is that in order to build a solver class only the `optimize()` method
    has to be implemented, that determines the "optimal" value for the optimization variable given its initial value
    and the internally stored scene, while the general solver implementation deals with multi-threading and provides
    methods for computing the objective and constraint values (given a list of modules which should be taken into
    account, see below).

    Initialise solver class by building objective and constraint modules as defined within the specific
    definition of the (optimisation) problem. Over all definitions it is shared that the optimization
    variable are the control inputs of the robot, however the exact formulation, i.e. which objective
    and constraint modules are used depends on the implementation of the child class.

    Internally, the solver stores two environments, the environment it uses for planning (optimization etc) and
    the environment it uses for evaluation, i.e. which is actually unknown for the solver but encodes the way
    the scene actually changes from one time-step to another. If `eval_env = None` the planning and evaluation
    environment are the same.

    :param env: environment the solver's forward simulations are based on.
    :param goal: goal state (position) of the robot (2).
    :param t_planning: planning horizon, i.e. how many future time-steps shall be taken into account in planning.
    :param multiprocessing: use multiprocessing for optimization.
    :param modules: List of optimization modules and according kwargs (if required).
    :param attention_module: Filter module name (None = no filter).
    :param eval_env: environment that should be used for evaluation ("real" environment).
    :param config_name: name of solver configuration.
    :param is_logging: should all the results be logged (necessary for plotting but very costly !!).
    :param is_debug: logging debug mode (for printing).
    """
    def __init__(
        self,
        env: mantrap.environment.base.GraphBasedEnvironment,
        goal: torch.Tensor,
        t_planning: int = mantrap.constants.SOLVER_HORIZON_DEFAULT,
        modules: typing.Union[typing.List[typing.Tuple], typing.List] = None,
        attention_module: mantrap.attention.AttentionModule.__class__ = None,
        eval_env: mantrap.environment.base.GraphBasedEnvironment = None,
        config_name: str = mantrap.constants.CONFIG_UNKNOWN,
        is_logging: bool = False,
        is_debug: bool = False,
        **solver_params
    ):
        # Dictionary of solver parameters.
        self._solver_params = solver_params
        self._solver_params[mantrap.constants.PK_T_PLANNING] = t_planning
        self._solver_params[mantrap.constants.PK_CONFIG] = config_name

        # Check and add goal state to solver's parameters.
        assert mantrap.utility.shaping.check_goal(goal)
        self._solver_params[mantrap.constants.PK_GOAL] = goal.detach().float()

        # Set planning and evaluation environment.
        self._env = env.copy()
        self._eval_env = eval_env.copy() if eval_env is not None else env.copy()
        assert self._env.same_initial_conditions(other=self._eval_env)
        assert self._env.ego is not None

        # The objective and constraint functions (and their gradients) are packed into modules, for a more compact
        # representation, the ease of switching between different functions and to simplify logging and
        # visualization.
        modules = self.module_defaults() if modules is None else modules
        self._module_dict = {}
        for module_tuple in modules:
            if type(module_tuple) == tuple:
                assert 1 <= len(module_tuple) <= 2
                module = module_tuple[0]
                module_kwargs = {} if len(module_tuple) < 2 else module_tuple[1]
                module_kwargs = {} if module_kwargs is None else module_kwargs
            else:
                module = module_tuple
                module_kwargs = {}
            module_object = module(t_horizon=self.planning_horizon, goal=self.goal, env=self.env, **module_kwargs)
            self._module_dict[module_object.name] = module_object

        # Attention module for "importance" selection of which ados to include into optimization.
        self._attention_module = None
        if attention_module is not None:
            self._attention_module = attention_module(env=self.env, t_horizon=self.planning_horizon)

        # Initialize logging class.
        self._logger = OptimizationLogger(is_logging=is_logging, is_debug=is_debug)
        self.logger.log_reset()

        # Sanity checks.
        assert self.num_optimization_variables() > 0
        self.env.sanity_check(check_ego=True)

    ###########################################################################
    # Solving #################################################################
    ###########################################################################
    def solve(self, time_steps: int, warm_start_method: str = mantrap.constants.WARM_START_HARD, **kwargs
              ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Find the ego trajectory given the internal environment with the current scene as initial condition.
        Therefore iteratively solve the problem for the scene at t = t_k, update the scene using the internal simulator
        and the derived ego policy and repeat until t_k = `horizon` or until the goal has been reached.

        This method changes the internal environment by forward simulating it over the prediction horizon.

        :param time_steps: how many time-steps shall be solved (not planning horizon !).
        :param warm_start_method: warm-starting method (see .warm_start()).
        :return: derived ego trajectory [horizon + 1, 5].
        :return: derived actual ado trajectories [num_ados, 1, horizon + 1, 5].
        """
        ego_trajectory_opt = torch.zeros((time_steps + 1, 5))
        ado_trajectories = torch.zeros((self.env.num_ados, time_steps + 1, 1, 5))
        self.logger.log_reset()
        env_copy = self.env.copy()
        eval_env_copy = self.eval_env.copy()

        # Initialize trajectories with current state and environment time.
        ego_trajectory_opt[0] = self._env.ego.state_with_time
        for m_ado, ado in enumerate(self.env.ados):
            ado_trajectories[m_ado, 0, 0, :] = ado.state_with_time

        # Warm-start the optimization using a simplified optimization formulation.
        z_warm_start = self.warm_start(method=warm_start_method)

        logging.debug(f"Starting trajectory optimization solving for planning horizon {time_steps} steps ...")
        for k in range(time_steps):
            logging.debug("#" * 30 + f"solver {self.log_name} @k={k}: initializing optimization")

            # Solve optimisation problem.
            z_k, _ = self.optimize(z_warm_start, tag=mantrap.constants.TAG_OPTIMIZATION, **kwargs)
            ego_controls_k = self.z_to_ego_controls(z_k.detach().numpy())
            assert mantrap.utility.shaping.check_ego_controls(ego_controls_k, t_horizon=self.planning_horizon)

            # Warm-starting using the next optimization at time-step k+1 using the recent results k.
            # As we have proceeded one time-step, use the recent results for one-step ahead, and append
            # zero control actions to it.
            z_warm_start = self.ego_controls_to_z(torch.cat((ego_controls_k[1:, :], torch.zeros((1, 2)))))
            z_warm_start = torch.from_numpy(z_warm_start)  # detached !

            # Forward simulate environment.
            ado_states, ego_state = self._eval_env.step(ego_action=ego_controls_k[0, :])
            self._env.step_reset(ego_next=ego_state, ado_next=ado_states)
            ego_trajectory_opt[k + 1, :] = ego_state
            ado_trajectories[:, k + 1, 0, :] = ado_states

            # If the goal state has been reached, break the optimization loop (and shorten trajectories to
            # contain only states up to now (i.e. k + 1 optimization steps instead of max_steps).
            if torch.norm(ego_state[0:2] - self.goal) < mantrap.constants.SOLVER_GOAL_END_DISTANCE:
                ego_trajectory_opt = ego_trajectory_opt[:k + 2, :].detach()
                ado_trajectories = ado_trajectories[:, :k + 2, :, :].detach()

                # Log a last time in order to log the final state, after the environment has executed it
                # its update step. However since the controls have not changed, but still the planned
                # trajectories should  all have the same shape, the concatenate no action (zero controls).
                if self.logger.is_logging:
                    ego_controls = torch.cat((ego_controls_k[1:, :], torch.zeros((1, 2))))
                    ego_trajectory = self.env.ego.unroll_trajectory(ego_controls, dt=self.env.dt)
                    self.__intermediate_log(ego_trajectory=ego_trajectory)
                break

            # Increment solver iteration.
            self.logger.increment()

        # Update the logging with the actual ego and ado trajectories (currently only samples).
        actual_dict = {f"{mantrap.constants.LT_EGO}_actual": ego_trajectory_opt,
                       f"{mantrap.constants.LT_ADO}_actual": ado_trajectories}
        self.logger.log_append(**actual_dict, tag=mantrap.constants.TAG_OPTIMIZATION)

        # Cleaning up solver environment and summarizing logging.
        logging.debug(f"solver {self.log_name}: logging trajectory optimization")
        self.env.detach()  # detach environment from computation graph
        self.logger.log_store(csv_name=f"{self.log_name}.{self.env.log_name}")

        # Reset environment to initial state. Some modules are also connected to the old environment,
        # which has been forward predicted now. Reset these to the original environment copy.
        self._env = env_copy
        self._eval_env = eval_env_copy
        for module in self.modules:
            module.reset_env(env=self.env)

        logging.debug(f"solver {self.log_name}: finishing up optimization process")
        return ego_trajectory_opt, ado_trajectories

    ###########################################################################
    # Optimization ############################################################
    ###########################################################################
    def optimize(self, z0: torch.Tensor, tag: str, **kwargs) -> typing.Tuple[torch.Tensor, float]:
        """Optimization core wrapper function.

        Filter the agents by using the attention module, execute the optimization, log
        the results and return the optimization results.

        :param z0: initial value of optimization variable.
        :param tag: name of optimization call (name of the core).
        :param kwargs: additional arguments for optimization core function.
        """
        # Filter the important ghost indices from the current scene state.
        if self._attention_module is not None:
            ado_ids = self._attention_module.compute()
            logging.debug(f"solver [{tag}]: optimizing w.r.t. important ado ids = {ado_ids}")
        else:
            ado_ids = self.env.ado_ids  # all ado ids (not filtered)

        # Computation is done in `optimize_core()` class that is implemented in child class.
        z_opt, obj_opt, log_opt = self.optimize_core(z0, ado_ids=ado_ids, tag=tag, **kwargs)

        # Logging the optimization results.
        if self.logger.is_logging:
            ego_trajectory_k = self.z_to_ego_trajectory(z_opt.detach().numpy(), return_leaf=False)
            self.__intermediate_log(ego_trajectory=ego_trajectory_k, tag=tag)
            self.logger.log_update({key: x for key, x in log_opt.items()})

        return z_opt, obj_opt

    @abc.abstractmethod
    def optimize_core(self, z0: torch.Tensor, tag: str, ado_ids: typing.List[str], **kwargs
                      ) -> typing.Tuple[torch.Tensor, float, typing.Dict[str, torch.Tensor]]:
        """Optimization function for single core to find optimal z-vector.

        Given some initial value `z0` find the optimal allocation for z with respect to the internally defined
        objectives and constraints. This function is executed in every thread in parallel, for different initial
        values `z0`. To simplify optimization not all agents in the scene have to be taken into account during
        the optimization but only the ones with ids defined in `ado_ids`.

        ATTENTION: Since several `optimize_core()` calls are spawned in parallel, one for every process, but
        originating from the same solver class, the method should be self-contained. Hence, no internal
        variables should be updated, since this would lead to race conditions ! If class variables have
        to be altered and used within this function, then assign them to the process tag !

        :param z0: initial value of optimization variables.
        :param tag: name of optimization call (name of the core).
        :param ado_ids: identifiers of ados that should be taken into account during optimization.
        :returns: z_opt (optimal values of optimization variable vector)
                  objective_opt (optimal objective value)
                  optimization_log (logging dictionary for this optimization = self.log)
        """
        raise NotImplementedError

    ###########################################################################
    # Problem formulation - Warm-Starting #####################################
    ###########################################################################
    def warm_start(self, method: str = mantrap.constants.WARM_START_HARD) -> torch.Tensor:
        """Compute warm-start for optimization decision variables z.

        - hard: solve the same optimization process but use the hard optimization modules
                (interaction-un-related modules) only.

        - encoding:

        - soft: solve the same optimization process but use the hard optimization modules
                as well as a safety constraint.

        :param method: method to use.
        :return: initial z values.
        """
        logging.debug(f"solver [warm_start]: method = {method} starting ...")
        if method == mantrap.constants.WARM_START_HARD:
            z_warm_start = self._warm_start_optimization(modules=self.module_hard())
        elif method == mantrap.constants.WARM_START_ENCODING:
            z_warm_start = self._warm_start_encoding()
        elif method == mantrap.constants.WARM_START_SOFT:
            modules_soft = [*self.module_hard(), mantrap.modules.HJReachabilityModule]
            z_warm_start = self._warm_start_optimization(modules=modules_soft)
        else:
            raise ValueError(f"Invalid warm starting-method {method} !")
        logging.debug(f"solver [warm_start]: finished ...")
        return z_warm_start

    def _warm_start_optimization(self, modules: typing.Union[typing.List[typing.Tuple], typing.List]) -> torch.Tensor:
        """Warm-Starting by solving simplified optimization problem.

        In order to warm start the optimization solve the same optimization process but use the a part of the
        optimization modules (objectives and constraints) only. These optimization modules should
        be very efficient to solve, e.g. convex, not include the simulation model, etc., but still give
        a good guess for the final actual solution.

        :param modules: list of optimization modules that should be taken into account.
        """
        solver_part = self.__class__(env=self.env, goal=self.goal, modules=modules,
                                     t_planning=self.planning_horizon, config_name=self.config_name,
                                     is_logging=self.logger.is_logging)

        # As initial guess for this first optimization, without prior knowledge, going straight
        # from the current position to the goal with maximal control input is chosen.
        _, u_max = self.env.ego.control_limits()
        dx_goal = self.goal - self.env.ego.position
        dx_goal_length = torch.norm(dx_goal).item()
        ego_controls_init = torch.stack([dx_goal / dx_goal_length * u_max] * self.planning_horizon)
        z_init = self.ego_controls_to_z(ego_controls=ego_controls_init)

        # Solve the simplified optimization and return its results.
        z_opt_hard, _ = solver_part.optimize(z0=torch.from_numpy(z_init), tag=mantrap.constants.TAG_WARM_START)
        self.logger.log_update(solver_part.logger.log)
        return z_opt_hard

    def _warm_start_encoding(self) -> torch.Tensor:
        raise NotImplementedError

    ###########################################################################
    # Problem formulation - Formulation #######################################
    ###########################################################################
    @staticmethod
    def module_defaults() -> typing.Union[typing.List[typing.Tuple], typing.List]:
        """List of optimization modules (objectives, constraint) and according dictionary
        of module kwargs, such as weight (for objectives), etc."""
        return [(mantrap.modules.GoalNormModule, {"optimize_speed": False, "weight": 1.0}),
                (mantrap.modules.InteractionProbabilityModule, {"weight": 1.0}),
                mantrap.modules.SpeedLimitModule,
                mantrap.modules.HJReachabilityModule]

    @staticmethod
    def module_hard() -> typing.Union[typing.List[typing.Tuple], typing.List]:
        """List of "hard" optimization modules (objectives, constraint). Hard modules are used for
        warm-starting the trajectory optimization and should therefore be simple to solve while still
        encoding a good guess of possible solutions.

        By default these modules are assumed to be the goal objective function and the controls limit
        constraint, dynamics constraint fulfilled by the solver's structure.
        """
        return [mantrap.modules.GoalNormModule, mantrap.modules.ControlLimitModule, mantrap.modules.SpeedLimitModule]

    def num_optimization_variables(self) -> int:
        return 2 * self.planning_horizon

    def optimization_variable_bounds(self) -> typing.Tuple[typing.List, typing.List]:
        lower, upper = self._env.ego.control_limits()
        lb = (np.ones(self.num_optimization_variables()) * lower).tolist()
        ub = (np.ones(self.num_optimization_variables()) * upper).tolist()
        return lb, ub

    ###########################################################################
    # Problem formulation - Objective #########################################
    ###########################################################################
    def objective(self, z: np.ndarray, ado_ids: typing.List[str] = None, tag: str = mantrap.constants.TAG_OPTIMIZATION
                  ) -> float:
        """Determine objective value for some optimization vector `z`.

        This function generally defines the value of the objective function for some input optimization vector `z`,
        and some input list of ados that should be taken into account in this computation. Therefore all objective
        modules are called, their results are summed (module-internally weighted).

        :param z: optimization vector (shape depends on exact optimization formulation).
        :param ado_ids: identifiers of ados that should be taken into account during optimization.
        :param tag: name of optimization call (name of the core).
        :return: weighted sum of objective values w.r.t. `z`.
        """
        ado_ids = ado_ids if ado_ids is not None else self.env.ado_ids
        ego_trajectory = self.z_to_ego_trajectory(z)
        objective = np.sum([m.objective(ego_trajectory, ado_ids=ado_ids, tag=tag) for m in self.modules])

        if self.logger.is_logging:
            module_log = {f"{mantrap.constants.LT_OBJECTIVE}_{key}": mod.obj_current(tag=tag)
                          for key, mod in self.module_dict.items()}
            module_log[f"{mantrap.constants.LT_OBJECTIVE}_{mantrap.constants.LK_OVERALL}"] = objective
            self.logger.log_append(**module_log, tag=tag)

        return float(objective)

    ###########################################################################
    # Problem formulation - Constraints #######################################
    ###########################################################################
    def constraints(
        self,
        z: np.ndarray,
        ado_ids: typing.List[str] = None,
        tag: str = mantrap.constants.TAG_OPTIMIZATION,
        return_violation: bool = False
    ) -> np.ndarray:
        """Determine constraints vector for some optimization vector `z`.

        This function generally defines the constraints vector for some input optimization vector `z`, and some
        list of ados that should be taken into account in this computation. Therefore all constraint modules
        are called, their results are concatenated.

        For logging purposes additionally the overall constraints violation, i.e. a scalar value representing the
        amount of how much each constraint is violated (zero if constraint is not active), summed over all
        constraints, is determined and logged, together with the violation by constraint module. Since the
        optimization parameters itself already have been logged in the `objective()` method, there are not logged
        within the `constraints()` method.

        If the optimization is unconstrained, an empty constraint vector as well as zero violation are returned.

        :param z: optimization vector (shape depends on exact optimization formulation).
        :param ado_ids: identifiers of ados that should be taken into account during optimization.
        :param tag: name of optimization call (name of the core).
        :param return_violation: flag whether to return the overall constraint violation value as well.
        :return: constraints vector w.r.t. `z`.
        """
        ado_ids = ado_ids if ado_ids is not None else self.env.ado_ids
        ego_trajectory = self.z_to_ego_trajectory(z)
        constraints = np.concatenate([m.constraint(ego_trajectory, tag=tag, ado_ids=ado_ids) for m in self.modules])
        violation = float(np.sum([m.compute_violation_internal(tag=tag) for m in self.modules]))

        if self.logger.is_logging:
            module_log = {f"{mantrap.constants.LT_CONSTRAINT}_{key}": mod.inf_current(tag=tag)
                          for key, mod in self.module_dict.items()}
            module_log[f"{mantrap.constants.LT_CONSTRAINT}_{mantrap.constants.LK_OVERALL}"] = violation
            self.logger.log_append(**module_log, tag=tag)

        return constraints if not return_violation else (constraints, violation)

    ###########################################################################
    # Transformations - Optimization variable z = control inputs u_t [0, T] ###
    ###########################################################################
    def z_to_ego_trajectory(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        ego_controls = torch.from_numpy(z).view(-1, 2).float()
        ego_controls.requires_grad = True
        ego_trajectory = self.env.ego.unroll_trajectory(controls=ego_controls, dt=self.env.dt)
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, pos_and_vel_only=True)
        return ego_trajectory if not return_leaf else (ego_trajectory, ego_controls)

    @staticmethod
    def z_to_ego_controls(z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        ego_controls = torch.from_numpy(z).view(-1, 2).float()
        ego_controls.requires_grad = True
        assert mantrap.utility.shaping.check_ego_controls(ego_controls)
        return ego_controls if not return_leaf else (ego_controls, ego_controls)

    def ego_trajectory_to_z(self, ego_trajectory: torch.Tensor) -> np.ndarray:
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory)
        controls = self.env.ego.roll_trajectory(ego_trajectory, dt=self.env.dt)
        return controls.flatten().detach().numpy()

    @staticmethod
    def ego_controls_to_z(ego_controls: torch.Tensor) -> np.ndarray:
        assert mantrap.utility.shaping.check_ego_controls(ego_controls)
        return ego_controls.flatten().detach().numpy()

    ###########################################################################
    # Encoding ################################################################
    ###########################################################################
    def encode(self) -> torch.Tensor:
        """Encode the current environment-goal-setup in a four-dimensional continuous space.

        To represent a given scene completely the following elements have to be taken into
        account: the robot state (position, acceleration, type),  the ado states (position,
        velocity, history, type) and the goal state. However to reduce dimensionality of
        the representation, we simplify the problem to only encode the current states (no
        history) and assume single integrator ado and double integrator robot dynamics.
        As a consequence the ado's velocity can be ignored as well (since it can change
        instantly due to the single integrator dynamics).

        The most important information with respect to the trajectory optimization surely
        is the relative position of the goal state, in robot coordinates. Therefore as
        an encoding the ado positions are transformed into the coordinate system spanned
        by the line from the robot's to the goal position (and its orthogonal). The
        transformed coordinates of the closest pedestrian (w.r.t. the robot) as well as
        the robot's velocity form the scene encoding.

        .. math::\\vec{s} = \\begin{bmatrix} \\eta_P & \\mu_P & vx_R & vy_R \\end{bmatrix}^T

        :return: four-dimensional scene representation.
        """
        with torch.no_grad():
            ego_state, ado_states = self.env.states()

            # Compute robot-goal-coordinate transformation.
            t = mantrap.utility.maths.rotation_matrix(ego_state[0:2], self.goal)

            # Determine closest pedestrian using L2-norm-distance.
            ado_distances = torch.norm(ado_states[:, 0:2] - ego_state[0:2], dim=1)
            i_ado_closest = torch.argmin(ado_distances)
            ado_pos_t = torch.matmul(t, ado_states[i_ado_closest, 0:2])

            return torch.cat((ado_pos_t, ego_state[2:4]))

    ###########################################################################
    # Logging #################################################################
    ###########################################################################
    def __intermediate_log(self, ego_trajectory: torch.Tensor, tag: str = mantrap.constants.TAG_OPTIMIZATION):
        if self.logger.is_logging:
            ado_planned = self.env.sample_w_trajectory(ego_trajectory=ego_trajectory, num_samples=10)
            ado_planned_wo = self.env.sample_wo_ego(t_horizon=ego_trajectory.shape[0] - 1, num_samples=10)
            trajectory_log = {f"{mantrap.constants.LT_EGO}_planned": ego_trajectory,
                              f"{mantrap.constants.LT_ADO}_planned": ado_planned,
                              f"{mantrap.constants.LT_ADO_WO}_planned": ado_planned_wo}
            self.logger.log_append(**trajectory_log, tag=tag)

    ###########################################################################
    # Visualization ###########################################################
    ###########################################################################
    def visualize_scenes(self, tag: str = mantrap.constants.TAG_OPTIMIZATION, **vis_kwargs):
        """Visualize planned trajectory over full time-horizon as well as simulated ado reactions (i.e. their
        trajectories conditioned on the planned ego trajectory).

        :param tag: logging tag to plot, per default optimization tag.
        """
        from mantrap.visualization.atomics import output_format
        from mantrap.visualization import visualize_optimization

        ego_planned = self.logger.log_query("planned", mantrap.constants.LT_EGO, tag=tag, apply_func="cat")
        ado_actual = self.logger.log_query("actual", mantrap.constants.LT_ADO, tag=tag, apply_func="cat")
        ado_planned = self.logger.log_query("planned", mantrap.constants.LT_ADO, tag=tag, apply_func="cat")
        ado_planned_wo = self.logger.log_query("planned", mantrap.constants.LT_ADO_WO, tag=tag, apply_func="cat")

        return visualize_optimization(
            ego_planned=ego_planned,
            ado_actual=ado_actual,
            ado_planned=ado_planned,
            ado_planned_wo=ado_planned_wo,
            # ego_trials=[self._log[f"{tag}/ego_planned_{k}"] for k in range(self._iteration + 1)],
            ego_goal=self.goal,
            env=self.env,
            file_path=output_format(f"{self.log_name}_{self.env.name}_scenes"),
            **vis_kwargs
        )

    def visualize_step(self, tag: str = mantrap.constants.TAG_OPTIMIZATION, iteration: int = 0, **vis_kwargs):
        """Visualize planned trajectory over full time-horizon as well as simulated ado reactions (i.e. their
        trajectories conditioned on the planned ego trajectory) for one specific iteration.

        :param tag: logging tag to plot, per default optimization tag.
        :param iteration: solver iteration to visualize, per default 0th iteration (start).
        """
        from mantrap.visualization import visualize_prediction
        from mantrap.visualization.atomics import output_format

        iter_string = str(iteration)
        lt_ego, lt_ado = mantrap.constants.LT_EGO, mantrap.constants.LT_ADO
        ego_planned = self.logger.log_query("planned", lt_ego, tag=tag, apply_func="cat", iteration=iter_string)
        ado_planned = self.logger.log_query("planned", lt_ado, tag=tag, apply_func="cat", iteration=iter_string)
        ado_planned_wo = self.logger.log_query("planned", lt_ado, tag=tag, apply_func="cat", iteration=iter_string)

        return visualize_prediction(
            ego_planned=ego_planned,
            ado_planned=ado_planned,
            ado_planned_wo=ado_planned_wo,
            ego_goal=self.goal,
            env=self.env,
            file_path=output_format(name=f"{self.log_name}_{self.env.name}_scenes"),
            **vis_kwargs
        )

    ###########################################################################
    # Solver parameters #######################################################
    ###########################################################################
    @property
    def env(self) -> mantrap.environment.base.GraphBasedEnvironment:
        return self._env

    @env.setter
    def env(self, env: mantrap.environment.base.GraphBasedEnvironment):
        self._env = env

    @property
    def eval_env(self) -> mantrap.environment.base.GraphBasedEnvironment:
        return self._eval_env

    @property
    def goal(self) -> torch.Tensor:
        return self._solver_params[mantrap.constants.PK_GOAL]

    @property
    def planning_horizon(self) -> int:
        return self._solver_params[mantrap.constants.PK_T_PLANNING]

    ###########################################################################
    # Optimization formulation parameters #####################################
    ###########################################################################
    @property
    def module_dict(self) -> typing.Dict[str, mantrap.modules.base.OptimizationModule]:
        return self._module_dict

    @property
    def modules(self) -> typing.List[mantrap.modules.base.OptimizationModule]:
        return list(self.module_dict.values())

    @property
    def module_names(self) -> typing.List[str]:
        return [mantrap.constants.LK_OVERALL] + list(self.module_dict.keys())

    @property
    def attention_module(self) -> str:
        return self._attention_module.name() if self._attention_module is not None else "none"

    ###########################################################################
    # Logging parameters ######################################################
    ###########################################################################
    @property
    def logger(self) -> OptimizationLogger:
        return self._logger
    
    @property
    def config_name(self) -> str:
        return self._solver_params[mantrap.constants.PK_CONFIG]

    @property
    def log_name(self) -> str:
        return self.name + "_" + self.config_name

    ###########################################################################
    # Solver properties #######################################################
    ###########################################################################
    @property
    def name(self) -> str:
        raise NotImplementedError
