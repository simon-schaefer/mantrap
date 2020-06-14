import abc
import collections
import logging
import os
import typing

import numpy as np
import pandas
import torch

import mantrap.constants
import mantrap.environment
import mantrap.attention
import mantrap.modules
import mantrap.utility


class TrajOptSolver(abc.ABC):
    """General abstract solver implementation.

    The idea of this general implementation is that in order to build a solver class only the `optimize()` method
    has to be implemented, that determines the "optimal" value for the optimization variable given its initial value
    and the internally stored scene, while the general solver implementation deals with multi-threading and provides
    methods for computing the objective and constraint values (given a list of modules which should be taken into
    account, see below).

    Initialise solver class by building objective and constraint modules as defined within the specific
    definition of the (optimisation) problem.

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

        # Logging variables. Using default-dict(deque) whenever a new entry is created, it does not have to be checked
        # whether the related key is already existing, since if it is not existing, it is created with a queue as
        # starting value, to which the new entry is appended. With an appending complexity O(1) instead of O(N) the
        # deque is way more efficient than the list type for storing simple floating point numbers in a sequence.
        self._log = None
        self._iteration = None
        self._is_logging = is_logging

        # Initialize child class.
        self.initialize(**solver_params)

        # Sanity checks.
        assert self.num_optimization_variables() > 0

    def initialize(self, **solver_params):
        """Method can be overwritten when further initialization is required."""
        pass

    @classmethod
    def solver_hard(
        cls,
        env: mantrap.environment.base.GraphBasedEnvironment,
        goal: torch.Tensor,
        t_planning: int = mantrap.constants.SOLVER_HORIZON_DEFAULT,
        config_name: str = mantrap.constants.CONFIG_UNKNOWN,
        **solver_params
    ):
        """Create internal solver version with only "hard" optimization modules."""
        modules_hard = cls.module_hard()
        return cls(env, goal, t_planning=t_planning, modules=modules_hard, config_name=config_name, **solver_params)

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
        self._log_reset()
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
            self._iteration = k

            # Solve optimisation problem.
            z_k, _, optimization_log = self.optimize(z_warm_start, tag=mantrap.constants.TAG_OPTIMIZATION, **kwargs)
            ego_controls_k = self.z_to_ego_controls(z_k.detach().numpy())
            assert mantrap.utility.shaping.check_ego_controls(ego_controls_k, t_horizon=self.planning_horizon)

            # Warm-starting using the next optimization at time-step k+1 using the recent results k.
            # As we have proceeded one time-step, use the recent results for one-step ahead, and append
            # zero control actions to it.
            z_warm_start = self.ego_controls_to_z(torch.cat((ego_controls_k[1:, :], torch.zeros((1, 2)))))
            z_warm_start = torch.from_numpy(z_warm_start)  # detached !

            # Logging, before the environment step is done and update optimization
            # logging values for optimization results.
            if self.is_logging:
                ego_trajectory_k = self.env.ego.unroll_trajectory(ego_controls_k, dt=self.env.dt)
                self.__intermediate_log(ego_trajectory=ego_trajectory_k)
                self._log.update({key: x for key, x in optimization_log.items()})

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
                if self.is_logging:
                    ego_controls = torch.cat((ego_controls_k[1:, :], torch.zeros((1, 2))))
                    ego_trajectory = self.env.ego.unroll_trajectory(ego_controls, dt=self.env.dt)
                    self.__intermediate_log(ego_trajectory=ego_trajectory)
                break

        # Update the logging with the actual ego and ado trajectories (currently only samples).
        actual_dict = {f"{mantrap.constants.LT_EGO}_actual": ego_trajectory_opt,
                       f"{mantrap.constants.LT_ADO}_actual": ado_trajectories}
        self._log_append(**actual_dict, tag=mantrap.constants.TAG_OPTIMIZATION)

        # Cleaning up solver environment and summarizing logging.
        logging.debug(f"solver {self.log_name}: logging trajectory optimization")
        self.env.detach()  # detach environment from computation graph
        self.__log_summarize()

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
    def optimize(self, z0: torch.Tensor, tag: str, **kwargs
                 ) -> typing.Tuple[torch.Tensor, float, typing.Dict[str, torch.Tensor]]:
        # Filter the important ghost indices from the current scene state.
        if self._attention_module is not None:
            ado_ids = self._attention_module.compute()
            logging.debug(f"solver [{tag}]: optimizing w.r.t. important ado ids = {ado_ids}")
        else:
            ado_ids = self.env.ado_ids  # all ado ids (not filtered)

        # Computation is done in `optimize_core()` class that is implemented in child class.
        return self.optimize_core(z0, ado_ids=ado_ids, tag=tag, **kwargs)

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

        - hard: solve the same optimization process but use the hard optimization modules (
                objectives and constraints) only.

        - encoding:

        :param method: method to use.
        :return: initial z values.
        """
        logging.debug(f"solver [warm_start]: method = {method} starting ...")
        if method == mantrap.constants.WARM_START_HARD:
            z_warm_start = self._warm_start_hard()
        elif method == mantrap.constants.WARM_START_ENCODING:
            z_warm_start = self._warm_start_encoding()
        else:
            raise ValueError(f"Invalid warm starting-method {method} !")
        logging.debug(f"solver [warm_start]: finished ...")
        return z_warm_start

    def _warm_start_hard(self) -> torch.Tensor:
        """Warm-Starting optimization using solution for hard modules only.

        In order to warm start the optimization solve the same optimization process but use the hard
        optimization modules (objectives and constraints) only. These hard optimization modules should
        be very efficient to solve, e.g. convex, not include the simulation model, etc., but still give
        a good guess for the final actual solution, i.e. the solution including soft modules as well.

        For further information about hard modules please have a look into `module_hard()`.
        """
        solver_hard = self.solver_hard(env=self.env, goal=self.goal,
                                       t_planning=self.planning_horizon, config_name=self.config_name)

        # As initial guess for this first optimization, without prior knowledge, going straight
        # from the current position to the goal with maximal control input is chosen.
        _, u_max = self.env.ego.control_limits()
        dx_goal = self.goal - self.env.ego.position
        dx_goal_length = torch.norm(dx_goal).item()
        ego_controls_init = torch.stack([dx_goal / dx_goal_length * u_max] * self.planning_horizon)
        z_init = self.ego_controls_to_z(ego_controls=ego_controls_init)

        # Solve the simplified optimization and return its results.
        z_opt_hard, _, _ = solver_hard.optimize(z0=torch.from_numpy(z_init), tag=mantrap.constants.TAG_WARM_START)
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
        raise NotImplementedError

    @staticmethod
    def module_hard() -> typing.Union[typing.List[typing.Tuple], typing.List]:
        """List of "hard" optimization modules (objectives, constraint). Hard modules are used for
        warm-starting the trajectory optimization and should therefore be simple to solve while still
        encoding a good guess of possible solutions.

        By default these modules are assumed to be the goal objective function and the controls limit
        constraint, dynamics constraint fulfilled by the solver's structure.
        """
        return [mantrap.modules.GoalNormModule,
                mantrap.modules.ControlLimitModule,
                mantrap.modules.SpeedLimitModule]

    @abc.abstractmethod
    def num_optimization_variables(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def optimization_variable_bounds(self) -> typing.Tuple[typing.List, typing.List]:
        raise NotImplementedError

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

        if self.is_logging:
            module_log = {f"{mantrap.constants.LT_OBJECTIVE}_{key}": mod.obj_current(tag=tag)
                          for key, mod in self.module_dict.items()}
            module_log[f"{mantrap.constants.LT_OBJECTIVE}_{mantrap.constants.LK_OVERALL}"] = objective
            self._log_append(**module_log, tag=tag)

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

        if self.is_logging:
            module_log = {f"{mantrap.constants.LT_CONSTRAINT}_{key}": mod.inf_current(tag=tag)
                          for key, mod in self.module_dict.items()}
            module_log[f"{mantrap.constants.LT_CONSTRAINT}_{mantrap.constants.LK_OVERALL}"] = violation
            self._log_append(**module_log, tag=tag)

        return constraints if not return_violation else (constraints, violation)

    ###########################################################################
    # Transformations #########################################################
    ###########################################################################
    @abc.abstractmethod
    def z_to_ego_trajectory(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def z_to_ego_controls(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def ego_trajectory_to_z(self, ego_trajectory: torch.Tensor) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def ego_controls_to_z(self, ego_controls: torch.Tensor) -> np.ndarray:
        raise NotImplementedError

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
        if self.is_logging and self.log is not None:
            ado_planned = self.env.sample_w_trajectory(ego_trajectory=ego_trajectory, num_samples=10)
            ado_planned_wo = self.env.sample_wo_ego(t_horizon=ego_trajectory.shape[0] - 1, num_samples=10)
            trajectory_log = {f"{mantrap.constants.LT_EGO}_planned": ego_trajectory,
                              f"{mantrap.constants.LT_ADO}_planned": ado_planned,
                              f"{mantrap.constants.LT_ADO_WO}_planned": ado_planned_wo}
            self._log_append(**trajectory_log, tag=tag)

    def log_keys_performance(self, tag: str = mantrap.constants.TAG_OPTIMIZATION) -> typing.List[str]:
        objective_keys = [f"{tag}/{mantrap.constants.LT_OBJECTIVE}_{key}" for key in self.module_names]
        constraint_keys = [f"{tag}/{mantrap.constants.LT_CONSTRAINT}_{key}" for key in self.module_names]
        return objective_keys + constraint_keys

    def _log_reset(self):
        self._iteration = 0
        if self.is_logging:
            self._log = collections.defaultdict(list)

    def _log_append(self, tag: str = mantrap.constants.TAG_OPTIMIZATION, **kwargs):
        if self.is_logging and self.log is not None:
            for key, value in kwargs.items():
                if value is None:
                    x = None
                else:
                    x = torch.tensor(value) if type(value) != torch.Tensor else value.detach()
                self._log[f"{tag}/{key}_{self._iteration}"].append(x)

    def __log_summarize(self):
        """Summarize optimisation-step dictionaries to a single tensor per logging key, e.g. collapse all objective
        value tensors for k = 1, 2, ..., N to one tensor for the whole optimisation process.

        Attention: It is assumed that the last value of each series is the optimal value for this kth optimisation,
        e.g. the last value of the objective tensor `obj_overall` should be the smallest one. However it is hard to
        validate for the general logging key, therefore it is up to the user to implement it correctly.
        """
        if self.is_logging:
            assert self.log is not None

            # The log values have been added one by one during the optimization, so that they are lists
            # of tensors, stack them to a single tensor.
            for key, values in self.log.items():
                if type(values) == list and len(values) > 0 and all(type(x) == torch.Tensor for x in values):
                    self._log[key] = torch.stack(values)

            # Save the optimization performance for every optimization step into logging file. Since the
            # optimization log is `torch.Tensor` typed, it has to be mapped to a list of floating point numbers
            # first using the `map(dtype, list)` function.
            output_path = mantrap.constants.VISUALIZATION_DIRECTORY
            output_path = mantrap.utility.io.build_os_path(output_path, make_dir=True, free=False)
            output_path = os.path.join(output_path, f"{self.log_name}.{self.env.log_name}.logging.csv")
            csv_log_k_keys = [f"{key}_{k}" for key in self.log_keys_performance() for k in range(self._iteration + 1)]
            csv_log_k_keys += self.log_keys_performance()
            csv_log = {key: map(float, self.log[key]) for key in csv_log_k_keys if key in self.log.keys()}
            pandas.DataFrame.from_dict(csv_log, orient='index').to_csv(output_path)

    def log_query(self, key: str, key_type: str, iteration: str = "", tag: str = None, apply_func: str = "cat",
                  ) -> typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor], None]:
        """Query internal log for some value with given key (log-key-structure: {tag}/{key_type}_{key}).

         :param key: query key, e.g. name of objective module.
         :param key_type: type of query (-> `mantrap.constants.LK_`...).
         :param iteration: optimization iteration to search in, if None then no iteration (summarized value).
         :param tag: logging tag to search in.
         :param apply_func: stack/concatenate tensors or concatenate last elements of results
                            if multiple results (shapes not checked !!) ["stack", "cat", "last", "as_dict"].
         """
        if not self.is_logging:
            raise LookupError("For querying the `is_logging` flag must be activate before solving !")
        assert self.log is not None
        if iteration == "end":
            iteration = str(self._iteration)

        # Search in log for elements that satisfy the query and return as dictionary. For the sake of
        # runtime the elements are stored as torch tensors in the log, therefore stack and list them.
        results_dict = {}
        query = f"{key_type}_{key}_{iteration}"
        if tag is not None:
            query = f"{tag}/{query}"
        for key, values in self.log.items():
            if query not in key:
                continue
            results_dict[key] = values

        # If only one element is in the dictionary, return not the dictionary but the item itself.
        # Otherwise go through arguments one by one and apply them.
        if apply_func == "as_dict":
            return results_dict
        num_results = len(results_dict.keys())
        if num_results == 1:
            results = results_dict.popitem()[1]
        elif apply_func == "cat":
            results = torch.cat([results_dict[key] for key in sorted(results_dict.keys())])
        elif apply_func == "stack":
            results = torch.stack([results_dict[key] for key in sorted(results_dict.keys())], dim=0)
        elif apply_func == "last":
            results = torch.tensor([results_dict[key][-1] for key in sorted(results_dict.keys())])
        else:
            raise ValueError(f"Undefined apply function for log query {apply_func} !")
        return results.squeeze(dim=0)

    ###########################################################################
    # Visualization ###########################################################
    ###########################################################################
    def visualize_scenes(self, tag: str = mantrap.constants.TAG_OPTIMIZATION, **vis_keys):
        """Visualize planned trajectory over full time-horizon as well as simulated ado reactions (i.e. their
        trajectories conditioned on the planned ego trajectory).

        :param tag: logging tag to plot, per default optimization tag.
        """
        from mantrap.visualization.atomics import output_format
        from mantrap.visualization import visualize_optimization
        if not self.is_logging:
            raise LookupError("For visualization the `is_logging` flag must be activate before solving !")
        assert self.log is not None

        ego_planned = self.log_query(key_type=mantrap.constants.LT_EGO, key="planned", tag=tag, apply_func="cat")
        ado_actual = self.log_query(key_type=mantrap.constants.LT_ADO, key="actual", tag=tag, apply_func="cat")
        ado_planned = self.log_query(key_type=mantrap.constants.LT_ADO, key="planned", tag=tag, apply_func="cat")
        ado_planned_wo = self.log_query(key_type=mantrap.constants.LT_ADO_WO, key="planned", tag=tag, apply_func="cat")

        return visualize_optimization(
            ego_planned=ego_planned,
            ado_actual=ado_actual,
            ado_planned=ado_planned,
            ado_planned_wo=ado_planned_wo,
            # ego_trials=[self._log[f"{tag}/ego_planned_{k}"] for k in range(self._iteration + 1)],
            ego_goal=self.goal,
            env=self.env,
            file_path=output_format(f"{self.log_name}_{self.env.name}_scenes"),
            **vis_keys
        )

    def visualize_heat_map(self, propagation: str = "log", resolution: float = 0.1):
        """Visualize heat map of objective and constraint function for planned ego trajectory at initial
        state of the system.

        Therefore span a grid over the full (usually 2D) optimization state space and compute the objective
        value as well as the constraint violation for each grid point. Overlaying the grid point values
        computed for each optimization module creates the shown heat-map.

        The `propagation` mode determines the way how the z values are chosen over time. Plotting every
        possible state combination over the full planning horizon surely is in-feasible. Therefore there
        are several propagation modes possible:

        - "log": Choose the optimization values as it was chosen during the last optimization, i.e. the ones
                 stored in the internal logging. Hence, when plotting the kth heat-map, only the kth value
                 is sampled, while the (k-1) z-values before are used from the optimal trajectory in the logging.
                 Requires solver.solve() call in before !

        - "constant": Keep z-values constant over whole time-horizon, e.g. when the optimization variables are
                      the robot's controls than use the same action over the full time-horizon. Hence, the
                      optimization variable space has to be sampled only once per time-step.
        """
        from mantrap.visualization.atomics import output_format
        from mantrap.visualization import visualize_heat_map
        assert propagation in ["log", "constant"]

        num_time_steps = self.planning_horizon
        lower, upper = self.optimization_variable_bounds()
        assert len(lower) == len(upper) == 2 * num_time_steps  # 2D (!)

        # Create optimization variable mesh grid with given resolution.
        # Assumption: Same optimization variable bounds over full time horizon
        # and over all optimization variables, strong assumption but should hold
        # within this project (!).
        # + resolution since otherwise np.arange will stop at upper - resolution (!)
        x_grid, y_grid = np.meshgrid(np.arange(lower[0], upper[0] + resolution, step=resolution),
                                     np.arange(lower[1], upper[1] + resolution, step=resolution))
        points = np.stack((x_grid, y_grid)).transpose().reshape(-1, 2)
        grid_shape = x_grid.shape

        # Receive planned ego trajectories from log (propagation = log).
        z_values_prior = None
        if propagation == "log":
            assert self.log is not None
            ego_planned = self.log[f"{mantrap.constants.TAG_OPTIMIZATION}/ego_planned_end"]
            ego_planned = ego_planned[0, :, :]  # initial system state
            z_values_prior = self.ego_trajectory_to_z(ego_trajectory=ego_planned)
            assert z_values_prior.size == self.planning_horizon * 2  # 2D !

        # Iterate over all points, compute objective and constraint violation and write
        # both in images. For simplification take all ados into account for this  visualization,
        # by setting `ado_ids = None`.
        images = np.zeros((num_time_steps, *grid_shape))
        for t in range(num_time_steps):
            objective_values = np.zeros(grid_shape)
            constraint_values = np.zeros(grid_shape)
            for i, point in enumerate(points):
                ix = i % grid_shape[1]
                iy = i // grid_shape[1]

                # Use the chosen optimization variable state trajectory until the current state,
                # then add the current point to it and transform back to ego trajectory, which
                # can be used to determine the objective/constraint values.
                if propagation == "log":
                    zs = np.concatenate((z_values_prior[:t*2], point))  # cat flat z values to flat point
                elif propagation == "constant":
                    zs = np.array([point] * (t + 1)).flatten()
                else:
                    raise ValueError(f"Invalid propagation mode {propagation} !")

                # Transform optimization value to ego trajectory, which is required for objective and
                # constraint violation computations.
                ego_trajectory = self.z_to_ego_trajectory(zs)

                # Compute the objective/constraint values and add to images.
                ado_ids = self.env.ado_ids
                tag = mantrap.constants.TAG_VISUALIZATION
                objective = np.sum([m.objective(ego_trajectory, ado_ids=ado_ids, tag=tag)
                                    for m in self.modules])
                violation = np.sum([m.compute_violation(ego_trajectory, ado_ids=ado_ids, tag=tag)
                                    for m in self.modules])
                objective_values[ix, iy] = float(objective)
                constraint_values[ix, iy] = float(violation)

            # Merge objective and constraint values by setting all values in an infeasible region
            # (= constraint violation > 0) to nan value, that will be assigned to a special color
            # in the heat-map later on.
            objective_values[constraint_values > 0.0] = np.nan

            # Copy resulting objective values into results image.
            images[t, :, :] = objective_values.copy()

        # When known convert the optimized z-values to a usable shape. When not known simply do not plot
        # them in the heat-map (by passing `None`).
        zs = None
        if propagation == "log":
            zs = z_values_prior.reshape(-1, 2)

        # Find image bounds (both value and axes bounds).
        bounds = (lower[:2], upper[:2])  # two-dimensions
        if not np.all(np.isnan(images)):  # not  all values are non (i.e. infeasible)
            c_min, c_max = float(np.nanmin(images)), float(np.nanmax(images))
        else:
            c_min, c_max = -np.inf, np.inf

        # Finally draw all the created images in plot using the `visualize_heat_map` function
        # defined in the internal visualization package.
        path = output_format(name=f"{self.log_name}_{self.env.name}_heat_map")
        return visualize_heat_map(images, bounds=bounds, color_bounds=(c_min, c_max), choices=zs,
                                  resolution=resolution, title="optimization landscape",
                                  ax_labels=("z1", "z2"), file_path=path)

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
    def log(self) -> typing.Dict[str, typing.Union[torch.Tensor, typing.List[torch.Tensor]]]:
        return self._log

    @property
    def is_logging(self) -> bool:
        return self._is_logging

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
