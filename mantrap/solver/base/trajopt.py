import abc
import logging
import os
import typing

import joblib
import numpy as np
import pandas
import torch

import mantrap.constants
import mantrap.controller
import mantrap.environment
import mantrap.filter
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
    :param filter_module: Filter module name (None = no filter).
    :param eval_env: environment that should be used for evaluation ("real" environment).
    :param config_name: name of solver configuration.
    """
    def __init__(
        self,
        env: mantrap.environment.base.GraphBasedEnvironment,
        goal: torch.Tensor,
        t_planning: int = mantrap.constants.SOLVER_HORIZON_DEFAULT,
        modules: typing.List[typing.Tuple] = None,
        filter_module: mantrap.filter.FilterModule.__class__ = None,
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
            assert len(module_tuple) <= 2
            module = module_tuple[0]
            module_kwargs = {} if len(module_tuple) < 2 else module_tuple[1]
            module_kwargs = {} if module_kwargs is None else module_kwargs
            module_object = module(t_horizon=self.planning_horizon, goal=self.goal, env=self.env, **module_kwargs)
            self._module_dict[module_object.name] = module_object

        # Filter module for "importance" selection of which ados to include into optimization.
        self._filter_module = None
        if filter_module is not None:
            self._filter_module = filter_module(env=self.env, t_horizon=self.planning_horizon)

        # Logging variables. Using default-dict(deque) whenever a new entry is created, it does not have to be checked
        # whether the related key is already existing, since if it is not existing, it is created with a queue as
        # starting value, to which the new entry is appended. With an appending complexity O(1) instead of O(N) the
        # deque is way more efficient than the list type for storing simple floating point numbers in a sequence.
        self._log = None
        self._iteration = None
        self._core_opt = None

        # Initialize child class.
        self.initialize(**solver_params)

        # Sanity checks.
        assert self.num_optimization_variables() > 0

    def solve(self, time_steps: int, multiprocessing: bool = True, **solver_kwargs
              ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Find the ego trajectory given the internal environment with the current scene as initial condition.
        Therefore iteratively solve the problem for the scene at t = t_k, update the scene using the internal simulator
        and the derived ego policy and repeat until t_k = `horizon` or until the goal has been reached.

        This method changes the internal environment by forward simulating it over the prediction horizon.

        :param time_steps: how many time-steps shall be solved (not planning horizon !).
        :param multiprocessing: use multiple threads for solving with each initial value in parallel.
        :return: derived ego trajectory [horizon + 1, 5].
        :return: derived actual ado trajectories [num_ados, 1, horizon + 1, 5].
        """
        ego_trajectory_opt = torch.zeros((time_steps + 1, 5))
        ado_trajectories = torch.zeros((self.env.num_ados, 1, time_steps + 1, 5))
        self.log_reset(log_horizon=time_steps)
        env_copy = self.env.copy()
        eval_env_copy = self.eval_env.copy()

        # Initialize trajectories with current state and environment time.
        ego_trajectory_opt[0] = self._env.ego.state_with_time
        for ghost in self.env.ghosts:
            m_ado, m_mode = self.env.convert_ghost_id(ghost_id=ghost.id)
            ado_trajectories[m_ado, 0, 0, :] = ghost.agent.state_with_time

        logging.debug(f"Starting trajectory optimization solving for planning horizon {time_steps} steps ...")
        for k in range(time_steps):
            logging.debug("#" * 30 + f"solver {self.log_name} @k={k}: initializing optimization")
            self._iteration = k

            # Solve optimisation problem.
            ego_controls_k = self.determine_ego_controls(multiprocessing=multiprocessing, **solver_kwargs)
            logging.debug(f"solver {self.log_name} @k={k}: finishing optimization")

            # Logging, before the environment step is done.
            self.intermediate_log(ego_controls_k=ego_controls_k)

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
                self.intermediate_log(ego_controls_k=torch.cat((ego_controls_k[1:, :], torch.zeros((1, 2)))))
                break

        # Cleaning up solver environment and summarizing logging.
        logging.debug(f"solver {self.log_name}: logging trajectory optimization")
        self.env.detach()  # detach environment from computation graph
        self.log_summarize()

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
    def determine_ego_controls(self, multiprocessing: bool = True, **solver_kwargs) -> torch.Tensor:
        """Determine the ego control inputs for the internally stated problem and the current state of the environment.
        The implementation crucially depends on the solver class itself and is hence not implemented here.

        :param multiprocessing: use multiple threads for solving with each initial value in parallel.
        :return: ego_controls: control inputs of ego agent for whole planning horizon.
        :return: optimization dictionary of solution (containing objective, infeasibility scores, etc.).
        """
        logging.debug(f"solver: solving optimization problem in parallel = {multiprocessing}")

        # Find initial values for optimization variable and assign one to each core.
        z0s = self.initial_values()
        assert z0s.shape[0] == len(self.cores)
        initial_values = list(zip(z0s, self.cores))
        logging.debug(f"solver: initial values = {initial_values}")

        # Solve optimisation problem for each initial condition, either in multiprocessing or sequential.
        # Requiring shared memory allows to run code in optimized manner over multiple processes, e.g. by
        # sharing the "__debug__" flag between all processes. Also the processes do not change shared by
        # (or making a deepcopy), so copying to every process the whole memory is not efficient (neither
        # required at all).
        if multiprocessing:
            results = joblib.Parallel(n_jobs=8, require="sharedmem")(joblib.delayed(self.optimize)
                                                                     (z0, tag, **solver_kwargs)
                                                                     for z0, tag in initial_values)
        else:
            results = [self.optimize(z0, tag, **solver_kwargs) for z0, tag in initial_values]

        # Update optimization logging values for optimization results.
        if __debug__ is True:
            for i, (_, _, optimization_log) in enumerate(results):
                self._log.update({key: x for key, x in optimization_log.items() if self.cores[i] in key})

        # Return controls with minimal objective function result.
        index_best = int(np.argmin([obj for _, obj, _ in results]))
        z_opt_best, self._core_opt = results[index_best][0], self.cores[index_best]

        # Convert the resulting optimization variable to control inputs.
        ego_controls = self.z_to_ego_controls(z_opt_best.detach().numpy())
        assert mantrap.utility.shaping.check_ego_controls(ego_controls, t_horizon=self.planning_horizon)
        return ego_controls

    def optimize(self, z0: torch.Tensor, tag: str, **kwargs
                 ) -> typing.Tuple[torch.Tensor, float, typing.Dict[str, torch.Tensor]]:
        # Filter the important ghost indices from the current scene state.
        if self._filter_module is not None:
            ado_ids = self._filter_module.compute()
            logging.debug(f"solver [{tag}]: optimizing w.r.t. important ado ids = {ado_ids}")
        else:
            ado_ids = self.env.ado_ids  # all ado ids (not filtered)

        # Computation is done in `_optimize()` class that is implemented in child class.
        return self._optimize(z0, ado_ids=ado_ids, tag=tag, **kwargs)

    @abc.abstractmethod
    def _optimize(self, z0: torch.Tensor, tag: str, ado_ids: typing.List[str], **kwargs
                  ) -> typing.Tuple[torch.Tensor, float, typing.Dict[str, torch.Tensor]]:
        """Optimization function for single core to find optimal z-vector.

        Given some initial value `z0` find the optimal allocation for z with respect to the internally defined
        objectives and constraints. This function is executed in every thread in parallel, for different initial
        values `z0`. To simplify optimization not all agents in the scene have to be taken into account during
        the optimization but only the ones with ids defined in `ado_ids`.

        ATTENTION: Since several `_optimize()` calls are spawned in parallel, one for every process, but
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
    # Initialization ##########################################################
    ###########################################################################
    def initialize(self, **solver_params):
        """Method can be overwritten when further initialization is required."""
        pass

    def initial_values(self, just_one: bool = False) -> torch.Tensor:
        """Initialize with three primitives, going from the current ego position to the goal point, following
        square shapes. The middle one (index = 1) is a straight line, the other two have some curvature,
        one positive and the other one negative curvature. When just one initial trajectory should be returned,
        then return straight line trajectory.

        :param just_one: flag whether to return just one trajectory or multiple.
        """
        with torch.no_grad():
            ego_pos, goal = self.env.ego.position, self.goal  # local variables for speed up looping
            ego_distance = torch.norm(goal - ego_pos)  # distance between ego and goal
            ego_direction = goal - ego_pos / ego_distance  # ego-goal-direction vector
            normal = mantrap.utility.maths.normal_line(ego_pos, goal)  # normal line to vector from ego to goal

            # At first check whether goal and current state are actually the same, in case just return the
            # optimization variables representing no control actions.
            if ego_distance < 1e-3:
                z0 = self.ego_controls_to_z(ego_controls=torch.zeros((self.planning_horizon, 2)))
                z0s = [z0, z0, z0]  # three initial trajectories

            else:
                z0s = []
                for i, distance in enumerate([- ego_distance / 2, 0.0, ego_distance / 2.0]):
                    control_points = [ego_pos + ego_direction * ego_distance * f_point + f_distance * distance * normal
                                      for f_point, f_distance in [(0, 0), (0.25, 0.5), (0.5, 1), (0.75, 0.5), (1, 0.0)]]
                    control_points = torch.stack(control_points, dim=0)

                    # Spline interpolation to build "continuous" path.
                    reference_path = mantrap.utility.maths.spline_interpolation(control_points,
                                                                                num_samples=self.planning_horizon)

                    # Determine controls to track the given trajectory. Afterwards check whether the determined
                    # controls are feasible in terms of the control limits of the ego agent.
                    controls = mantrap.controller.p_ahead_controller(
                        agent=self.env.ego,
                        path=reference_path,
                        max_sim_time=self.planning_horizon * self.env.dt,
                        dtc=self.env.dt,
                        speed_reference=self.env.ego.speed_max
                    )

                    # If the controller did not need the full time until reaching the end of the reference path
                    # it has to be expanded (to be transformable to a valid optimization variable). Since the
                    # controller is assumed to break at the end, the easiest (and also valid) approach is to fill
                    # with zeros.
                    if controls.shape[0] < self.planning_horizon:
                        delta_horizon = self.planning_horizon - controls.shape[0]
                        delta_zeros = torch.zeros((delta_horizon, controls.shape[1]))
                        controls = torch.cat((controls, delta_zeros), dim=0)
                    assert mantrap.utility.shaping.check_ego_controls(controls, self.planning_horizon)
                    assert self.env.ego.check_feasibility_controls(controls)

                    # Transform controls to z variable.
                    z0s.append(self.ego_controls_to_z(controls))

        z0s = torch.from_numpy(np.array(z0s)).view(3, -1)
        logging.debug(f"solver: initial values z = {z0s}")
        return z0s if not just_one else z0s[1, :]

    @staticmethod
    def num_initial_values() -> int:
        return 3

    ###########################################################################
    # Problem formulation - Formulation #######################################
    ###########################################################################
    @abc.abstractmethod
    def num_optimization_variables(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def optimization_variable_bounds(self) -> typing.Tuple[typing.List, typing.List]:
        raise NotImplementedError

    @abc.abstractmethod
    def module_defaults(self) -> typing.List[typing.Tuple]:
        """List of optimization modules (objectives, constraint) and according dictionary
        of module kwargs, such as weight (for objectives), etc."""
        raise NotImplementedError

    ###########################################################################
    # Problem formulation - Objective #########################################
    ###########################################################################
    def objective(self, z: np.ndarray, ado_ids: typing.List[str] = None, tag: str = mantrap.constants.TAG_DEFAULT
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
        ego_trajectory = self.z_to_ego_trajectory(z)
        objective = np.sum([m.objective(ego_trajectory, ado_ids=ado_ids, tag=tag) for m in self.modules])

        if __debug__ is True:
            ado_planned = self.env.predict_w_trajectory(ego_trajectory=ego_trajectory)
            ado_planned_wo = self.env.predict_wo_ego(t_horizon=ego_trajectory.shape[0])
            self.log_append(ego_planned=ego_trajectory, ado_planned=ado_planned, ado_planned_wo=ado_planned_wo, tag=tag)
            self.log_append(obj_overall=objective, tag=tag)
            module_log = {f"{mantrap.constants.LK_OBJECTIVE}_{key}": mod.obj_current(tag=tag)
                          for key, mod in self.module_dict.items()}
            self.log_append(**module_log, tag=tag)

        return float(objective)

    ###########################################################################
    # Problem formulation - Constraints #######################################
    ###########################################################################
    def constraints(
        self,
        z: np.ndarray,
        ado_ids: typing.List[str] = None,
        tag: str = mantrap.constants.TAG_DEFAULT,
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
        ego_trajectory = self.z_to_ego_trajectory(z)
        constraints = np.concatenate([m.constraint(ego_trajectory, tag=tag, ado_ids=ado_ids) for m in self.modules])
        violation = float(np.sum([m.compute_violation_internal(tag=tag) for m in self.modules]))

        if __debug__ is True:
            self.log_append(inf_overall=violation, tag=tag)
            module_log = {f"{mantrap.constants.LK_CONSTRAINT}_{key}": mod.inf_current(tag=tag)
                          for key, mod in self.module_dict.items()}
            self.log_append(**module_log, tag=tag)

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
    def intermediate_log(self, ego_controls_k: torch.Tensor):
        if __debug__ is True and self.log is not None:
            # For logging purposes unroll and predict the scene for the derived ego controls.
            ego_opt_planned = self.env.ego.unroll_trajectory(controls=ego_controls_k, dt=self.env.dt)
            self.log_append(ego_planned=ego_opt_planned, tag=mantrap.constants.LK_OPTIMAL)
            ado_planned = self._env.predict_w_controls(ego_controls=ego_controls_k)
            self.log_append(ado_planned=ado_planned, tag=mantrap.constants.LK_OPTIMAL)
            ado_planned_wo = self._env.predict_wo_ego(t_horizon=ego_controls_k.shape[0] + 1)
            self.log_append(ado_planned_wo=ado_planned_wo, tag=mantrap.constants.LK_OPTIMAL)

    @staticmethod
    def log_keys() -> typing.List[str]:
        return ["ego_planned", "ado_planned", "ado_planned_wo"]

    def log_keys_performance(self) -> typing.List[str]:
        objective_keys = [f"{tag}/{mantrap.constants.LK_OBJECTIVE}_{key}"
                          for key in self.module_names for tag in self.cores]
        constraint_keys = [f"{tag}/{mantrap.constants.LK_CONSTRAINT}_{key}"
                           for key in self.module_names for tag in self.cores]
        return objective_keys + constraint_keys

    def log_keys_all(self) -> typing.List[str]:
        log_tag_keys = [f"{tag}/{key}" for key in self.log_keys() for tag in self.cores]
        log_opt_keys = [f"{mantrap.constants.LK_OPTIMAL}/{key}" for key in self.log_keys()]
        return self.log_keys_performance() + log_tag_keys + log_opt_keys

    def log_reset(self, log_horizon: int):
        # Reset iteration counter.
        self._iteration = 0

        # Reset optimization log by re-creating dictionary with entries all keys in the planning horizon. During
        # optimization new values are then added to these created lists.
        if __debug__ is True:
            self._log = {f"{tag}_{k}": [] for k in range(log_horizon) for tag in self.log_keys_all()}

    def log_append(self, tag: str = mantrap.constants.TAG_DEFAULT, **kwargs):
        if __debug__ is True and self.log is not None:
            for key, value in kwargs.items():
                x = torch.tensor(value) if type(value) != torch.Tensor else value.detach()
                self._log[f"{tag}/{key}_{self._iteration}"].append(x)

    def log_summarize(self):
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
                if len(summary) > 0:
                    self._log[key] = torch.stack(summary)

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

    ###########################################################################
    # Visualization ###########################################################
    ###########################################################################
    def visualize_scenes(self, plot_path_only: bool = False):
        """Visualize planned trajectory over full time-horizon as well as simulated ado reactions (i.e. their
        trajectories conditioned on the planned ego trajectory), if __debug__ is True (otherwise no logging). """
        if __debug__ is True:
            from mantrap.visualization import visualize_overview
            assert self.log is not None

            # From optimization log extract the core (initial condition) which has resulted in the best objective
            # value in the end. Then, due to the structure demanded by the visualization function, repeat the entry
            # N=t_horizon times to be able to visualize the whole distribution at every time.
            obj_dict = {key: self.log[f"{self.core_opt}/{mantrap.constants.LK_OBJECTIVE}_{key}"]
                        for key in self.module_names}
            obj_dict = {key: [obj_dict[key]] * (self._iteration + 1) for key in self.module_names}
            inf_dict = {key: self.log[f"{self.core_opt}/{mantrap.constants.LK_CONSTRAINT}_{key}"]
                        for key in self.module_names}
            inf_dict = {key: [inf_dict[key]] * (self._iteration + 1) for key in self.module_names}

            return visualize_overview(
                ego_planned=self.log[f"{mantrap.constants.LK_OPTIMAL}/ego_planned"],
                ado_planned=self.log[f"{mantrap.constants.LK_OPTIMAL}/ado_planned"],
                ado_planned_wo=self.log[f"{mantrap.constants.LK_OPTIMAL}/ado_planned_wo"],
                ego_trials=[self._log[f"{self.core_opt}/ego_planned_{k}"] for k in range(self._iteration + 1)],
                ego_goal=self.goal, obj_dict=obj_dict, inf_dict=inf_dict, env=self.env,
                plot_path_only=plot_path_only, file_path=self._visualize_output_format("scenes")
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
                ego_planned = self.log[f"{mantrap.constants.LK_OPTIMAL}/ego_planned"]
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
                    objective = np.sum([m.objective(ego_trajectory, ado_ids=ado_ids, tag="vis")
                                        for m in self.modules])
                    violation = np.sum([m.compute_violation(ego_trajectory, ado_ids=ado_ids, tag="vis")
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
            bounds = (lower[:2], upper[:2])
            return visualize_heat_map(images, bounds=bounds, choices=zs, resolution=resolution, file_path=path)

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
        return self._filter_module.name() if self._filter_module is not None else "none"

    ###########################################################################
    # Utility parameters ######################################################
    ###########################################################################
    @property
    def cores(self) -> typing.List[str]:
        return [f"core{i}" for i in range(self.num_initial_values())]

    @property
    def core_opt(self) -> str:
        return self._core_opt if self._core_opt is not None else mantrap.constants.LK_OPTIMAL

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
