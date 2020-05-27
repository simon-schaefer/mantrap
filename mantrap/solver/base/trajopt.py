import abc
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

        # Initialize child class.
        self.initialize(**solver_params)

        # Sanity checks.
        assert self.num_optimization_variables() > 0

    def initialize(self, **solver_params):
        """Method can be overwritten when further initialization is required."""
        pass

    ###########################################################################
    # Solving #################################################################
    ###########################################################################
    def solve(self, time_steps: int, **kwargs) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Find the ego trajectory given the internal environment with the current scene as initial condition.
        Therefore iteratively solve the problem for the scene at t = t_k, update the scene using the internal simulator
        and the derived ego policy and repeat until t_k = `horizon` or until the goal has been reached.

        This method changes the internal environment by forward simulating it over the prediction horizon.

        :param time_steps: how many time-steps shall be solved (not planning horizon !).
        :return: derived ego trajectory [horizon + 1, 5].
        :return: derived actual ado trajectories [num_ados, 1, horizon + 1, 5].
        """
        ego_trajectory_opt = torch.zeros((time_steps + 1, 5))
        ado_trajectories = torch.zeros((self.env.num_ados, 1, time_steps + 1, 5))
        self._log_reset(log_horizon=time_steps)
        env_copy = self.env.copy()
        eval_env_copy = self.eval_env.copy()

        # Initialize trajectories with current state and environment time.
        ego_trajectory_opt[0] = self._env.ego.state_with_time
        for ghost in self.env.ghosts:
            m_ado, m_mode = self.env.convert_ghost_id(ghost_id=ghost.id)
            ado_trajectories[m_ado, 0, 0, :] = ghost.agent.state_with_time

        # Warm-start the optimization using a simplified optimization formulation.
        z_warm_start = self.warm_start()

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
            self.__intermediate_log(ego_controls_k=ego_controls_k)
            if __debug__ is True:
                self._log.update({key: x for key, x in optimization_log.items()})

            # Forward simulate environment.
            ado_states, ego_state = self._eval_env.step(ego_action=ego_controls_k[0, :])
            self._env.step_reset(ego_state_next=ego_state, ado_states_next=ado_states)
            ego_trajectory_opt[k + 1, :] = ego_state
            ado_trajectories[:, 0, k + 1, :] = ado_states

            # If the goal state has been reached, break the optimization loop (and shorten trajectories to
            # contain only states up to now (i.e. k + 1 optimization steps instead of max_steps).
            if torch.norm(ego_state[0:2] - self.goal) < mantrap.constants.SOLVER_GOAL_END_DISTANCE:
                ego_trajectory_opt = ego_trajectory_opt[:k + 2, :].detach()
                ado_trajectories = ado_trajectories[:, :, :k + 2, :].detach()

                # Log a last time in order to log the final state, after the environment has executed it
                # its update step. However since the controls have not changed, but still the planned
                # trajectories should  all have the same shape, the concatenate no action (zero controls).
                self.__intermediate_log(ego_controls_k=torch.cat((ego_controls_k[1:, :], torch.zeros((1, 2)))))
                break

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
    def warm_start(self) -> torch.Tensor:
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

        Since the objective function is the only overlap between all solver classes, for logging purposes (and
        if the required verbosity level is met), after deriving the objective value the current optimization vector
        is logged, after being transformed into an understandable format (ego trajectory). Also the other parameters
        such as the objective values for every objective module are logged.

        :param z: optimization vector (shape depends on exact optimization formulation).
        :param ado_ids: identifiers of ados that should be taken into account during optimization.
        :param tag: name of optimization call (name of the core).
        :return: weighted sum of objective values w.r.t. `z`.
        """
        ado_ids = ado_ids if ado_ids is not None else self.env.ado_ids
        ego_trajectory = self.z_to_ego_trajectory(z)
        objective = np.sum([m.objective(ego_trajectory, ado_ids=ado_ids, tag=tag) for m in self.modules])

        if __debug__ is True:
            ado_planned = self.env.predict_w_trajectory(ego_trajectory=ego_trajectory)
            ado_planned_wo = self.env.predict_wo_ego(t_horizon=ego_trajectory.shape[0])
            self._log_append(tag, ego_planned=ego_trajectory, ado_planned=ado_planned, ado_planned_wo=ado_planned_wo)
            self._log_append(tag, obj_overall=objective)
            module_log = {f"{mantrap.constants.LT_OBJECTIVE}_{key}": mod.obj_current(tag=tag)
                          for key, mod in self.module_dict.items()}
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

        if __debug__ is True:
            self._log_append(inf_overall=violation, tag=tag)
            module_log = {f"{mantrap.constants.LT_CONSTRAINT}_{key}": mod.inf_current(tag=tag)
                          for key, mod in self.module_dict.items()}
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
    # Logging #################################################################
    ###########################################################################
    def __intermediate_log(self, ego_controls_k: torch.Tensor):
        if __debug__ is True and self.log is not None:
            # For logging purposes unroll and predict the scene for the derived ego controls.
            ego_opt_planned = self.env.ego.unroll_trajectory(controls=ego_controls_k, dt=self.env.dt)
            self._log_append(ego_planned=ego_opt_planned, tag=mantrap.constants.TAG_OPTIMIZATION)
            ado_planned = self._env.predict_w_controls(ego_controls=ego_controls_k)
            self._log_append(ado_planned=ado_planned, tag=mantrap.constants.TAG_OPTIMIZATION)
            ado_planned_wo = self._env.predict_wo_ego(t_horizon=ego_controls_k.shape[0] + 1)
            self._log_append(ado_planned_wo=ado_planned_wo, tag=mantrap.constants.TAG_OPTIMIZATION)

    @staticmethod
    def log_keys() -> typing.List[str]:
        return ["ego_planned", "ado_planned", "ado_planned_wo"]

    def log_keys_performance(self, tag: str = mantrap.constants.TAG_OPTIMIZATION) -> typing.List[str]:
        objective_keys = [f"{tag}/{mantrap.constants.LT_OBJECTIVE}_{key}" for key in self.module_names]
        constraint_keys = [f"{tag}/{mantrap.constants.LT_CONSTRAINT}_{key}" for key in self.module_names]
        return objective_keys + constraint_keys

    def log_keys_all(self, tag: str = mantrap.constants.TAG_OPTIMIZATION) -> typing.List[str]:
        return self.log_keys_performance(tag=tag) + [f"{tag}/{key}" for key in self.log_keys()]

    def _log_reset(self, log_horizon: int):
        # Reset iteration counter.
        self._iteration = 0

        # Reset optimization log by re-creating dictionary with entries all keys in the planning horizon. During
        # optimization new values are then added to these created lists.
        if __debug__ is True:
            self._log = {f"{key}_{k}": [] for k in range(log_horizon) for key in self.log_keys_all()}

    def _log_append(self, tag: str = mantrap.constants.TAG_OPTIMIZATION, **kwargs):
        if __debug__ is True and self.log is not None:
            for key, value in kwargs.items():
                x = torch.tensor(value) if type(value) != torch.Tensor else value.detach()
                self._log[f"{tag}/{key}_{self._iteration}"].append(x)

    def __log_summarize(self):
        """Summarize optimisation-step dictionaries to a single tensor per logging key, e.g. collapse all objective
        value tensors for k = 1, 2, ..., N to one tensor for the whole optimisation process.

        Attention: It is assumed that the last value of each series is the optimal value for this kth optimisation,
        e.g. the last value of the objective tensor `obj_overall` should be the smallest one. However it is hard to
        validate for the general logging key, therefore it is up to the user to implement it correctly.
        """
        if __debug__ is True:
            assert self.log is not None
            # Stack always the last values in the step-dictionaries (lists of logging values for each optimization
            # step), since it is assumed to be the most optimal one (e.g. for IPOPT).
            for key in self.log_keys_all():
                assert all([f"{key}_{k}" in self.log.keys() for k in range(self._iteration + 1)])
                summary = [self.log[f"{key}_{k}"][-1] for k in range(self._iteration + 1)
                           if len(self.log[f"{key}_{k}"]) > 0]
                summary = torch.stack(summary) if len(summary) > 0 else []
                self._log[f"{key}_end"] = summary

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

    def log_query(self, key: str, key_type: str, iteration: int = None, tag: str = None
                  ) -> typing.Union[typing.Dict[str, typing.List[float]], None]:
        """Query internal log for some value with given key (log-key-structure: {tag}/{key_type}_{key}).

         :param key: query key, e.g. name of objective module.
         :param key_type: type of query (-> mantrap.constants.LK_...).
         :param iteration: optimization iteration to search in, if None then no iteration (summarized value).
         :param tag: logging tag to search in.
         """
        assert self.log is not None
        assert iteration is None or iteration <= self._iteration

        # Build query by combining arguments into one query string.
        iteration = "end" if iteration is None else iteration
        query = f"{key_type}_{key}_{iteration}"
        if tag is not None:
            query = f"{tag}/{query}"

        # Search in log for elements that satisfy the query and return as dictionary. For the sake of
        # runtime the elements are stored as torch tensors in the log, therefore stack and list them.
        results_dict = {}
        for key, values in self.log.items():
            if query not in key:
                continue
            key_short = key.replace(f"_{iteration}", "")
            key_values = torch.stack(values).tolist() if type(values) == list else values.tolist()
            results_dict[key_short] = key_values
        return results_dict

    ###########################################################################
    # Visualization ###########################################################
    ###########################################################################
    def visualize_scenes(self, plot_path_only: bool = False, tag: str = mantrap.constants.TAG_OPTIMIZATION, **vis_keys):
        """Visualize planned trajectory over full time-horizon as well as simulated ado reactions (i.e. their
        trajectories conditioned on the planned ego trajectory), if __debug__ is True (otherwise no logging).

        :param plot_path_only: just plot the robot's and ado's trajectories, no further stats.
        :param tag: logging tag to plot, per default optimization tag.
        """
        if __debug__ is True:
            from mantrap.visualization import visualize_overview
            assert self.log is not None

            # From optimization log extract the core (initial condition) which has resulted in the best objective
            # value in the end. Then, due to the structure demanded by the visualization function, repeat the entry
            # N=t_horizon times to be able to visualize the whole distribution at every time.
            obj_dict = {key: self.log[f"{tag}/{mantrap.constants.LT_OBJECTIVE}_{key}_end"]
                        for key in self.module_names}
            obj_dict = {key: [obj_dict[key]] * (self._iteration + 1) for key in self.module_names}
            inf_dict = {key: self.log[f"{tag}/{mantrap.constants.LT_CONSTRAINT}_{key}_end"]
                        for key in self.module_names}
            inf_dict = {key: [inf_dict[key]] * (self._iteration + 1) for key in self.module_names}

            return visualize_overview(
                ego_planned=self.log[f"{tag}/ego_planned_end"],
                ado_planned=self.log[f"{tag}/ado_planned_end"],
                ado_planned_wo=self.log[f"{tag}/ado_planned_wo_end"],
                ego_trials=[self._log[f"{tag}/ego_planned_{k}"] for k in range(self._iteration + 1)],
                ego_goal=self.goal, obj_dict=obj_dict, inf_dict=inf_dict,
                env=self.env,
                plot_path_only=plot_path_only,
                file_path=self._visualize_output_format("scenes"),
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

        Plot only if __debug__ is True (otherwise no logging).
        """
        if __debug__ is True:
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

            # Finally draw all the created images in plot using the `visualize_heat_map` function
            # defined in the internal visualization package.
            path = self._visualize_output_format(name="heat_map")
            bounds = (lower[:2], upper[:2])  # two-dimensions
            c_min, c_max = float(np.nanmin(images)), float(np.nanmax(images))
            return visualize_heat_map(images, bounds=bounds, color_bounds=(c_min, c_max), choices=zs,
                                      resolution=resolution, title="optimization landscape",
                                      ax_labels=("z1", "z2"), file_path=path)

    def _visualize_output_format(self, name: str) -> typing.Union[str, None]:
        """The `visualize()` function enables interactive mode, i.e. returning the video as html5-video directly,
        # instead of saving it as ".gif"-file. Therefore depending on the input flags, set the output path
        # to None (interactive mode) or to an actual path (storing mode). """
        from mantrap.utility.io import build_os_path, is_running_from_ipython
        interactive = is_running_from_ipython()
        if not interactive:
            output_path = build_os_path(mantrap.constants.VISUALIZATION_DIRECTORY, make_dir=True, free=False)
            output_path = os.path.join(output_path, f"{self.log_name}.{self.env.log_name}_{name}")
        else:
            output_path = None
        return output_path

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
        return [mantrap.constants.LK_OVERALL_PERFORMANCE] + list(self.module_dict.keys())

    def filter_module(self) -> str:
        return self._attention_module.name() if self._attention_module is not None else "none"

    ###########################################################################
    # Logging parameters ######################################################
    ###########################################################################
    @property
    def log(self) -> typing.Dict[str, typing.Union[torch.Tensor, typing.List[torch.Tensor]]]:
        return self._log

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
