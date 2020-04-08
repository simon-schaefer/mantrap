from abc import abstractmethod
import logging
import os
from typing import Dict, List, Tuple, Union

import joblib
import numpy as np
import torch

from mantrap.constants import solver_horizon
from mantrap.environment.environment import GraphBasedEnvironment
from mantrap.solver.constraints.constraint_module import ConstraintModule
from mantrap.solver.constraints import CONSTRAINTS
from mantrap.solver.objectives.objective_module import ObjectiveModule
from mantrap.solver.objectives import OBJECTIVES
from mantrap.utility.io import build_os_path
from mantrap.utility.shaping import check_ego_controls


class Solver:
    """General abstract solver implementation.

    The idea of this general implementation is that in order to build a solver class only the `optimize()` method
    has to be implemented, that determines the "optimal" value for the optimization variable given its initial value
    and the internally stored scene, while the general solver implementation deals with multi-threading and provides
    methods for computing the objective and constraint values (given a list of modules which should be taken into
    account, see below).

    Initialise solver class by building objective and constraint modules as defined within the specific
    definition of the (optimisation) problem. The verbose flag enables printing more debugging flags as well
    as plotting the optimisation history at the end.

    Internally, the solver stores two environments, the environment it uses for planning (optimization etc) and
    the environment it uses for evaluation, i.e. which is actually unknown for the solver but encodes the way
    the scene actually changes from one time-step to another. If `eval_env = None` the planning and evaluation
    environment are the same.

    :param env: environment the solver's forward simulations are based on.
    :param goal: goal state (position) of the robot (2).
    :param t_planning: planning horizon, i.e. how many future time-steps shall be taken into account in planning.
    :param verbose: debugging flag (-1: nothing, 0: logging, 1: +printing, 2: +plot scenes, 3: +plot optimization).
    :param multiprocessing: use multiprocessing for optimization.
    :param objectives: List of objective module names and according weights.
    :param constraints: List of constraint module names.
    :param eval_env: environment that should be used for evaluation ("real" environment).
    :param config_name: name of solver configuration.
    """
    def __init__(
        self,
        env: GraphBasedEnvironment,
        goal: torch.Tensor,
        t_planning: int = solver_horizon,
        objectives: List[Tuple[str, float]] = None,
        constraints: List[str] = None,
        eval_env: GraphBasedEnvironment = None,
        verbose: int = -1,
        multiprocessing: bool = True,
        config_name: str = "unknown",
        **solver_params
    ):
        assert goal.size() == torch.Size([2])
        self._goal = goal.float()

        # Set planning and evaluation environment.
        self._env = env.copy()
        self._eval_env = eval_env.copy() if eval_env is not None else env.copy()
        assert self._env.same_initial_conditions(other=self._eval_env)

        # Dictionary of solver parameters.
        self._solver_params = solver_params
        self._solver_params["t_planning"] = t_planning
        self._solver_params["verbose"] = verbose
        self._solver_params["multiprocessing"] = multiprocessing

        # The objective and constraint functions (and their gradients) are packed into objectives, for a more compact
        # representation, the ease of switching between different objective functions and to simplify logging and
        # visualization.
        objective_modules = self.objective_defaults() if objectives is None else objectives
        self._objective_modules = self._build_objective_modules(modules=objective_modules)
        constraint_modules = self.constraints_defaults() if constraints is None else constraints
        self._constraint_modules = self._build_constraint_modules(modules=constraint_modules)

        # Logging variables. Using default-dict(deque) whenever a new entry is created, it does not have to be checked
        # whether the related key is already existing, since if it is not existing, it is created with a queue as
        # starting value, to which the new entry is appended. With an appending complexity O(1) instead of O(N) the
        # deque is way more efficient than the list type for storing simple floating point numbers in a sequence.
        self._optimization_log = None
        self._iteration = None
        self._core_opt = None
        self._config_name = config_name

        # Initialize child class.
        self.initialize(**solver_params)

        # Sanity checks.
        assert self.num_optimization_variables() > 0

    def solve(self, time_steps: int, **solver_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find the ego trajectory given the internal environment with the current scene as initial condition.
        Therefore iteratively solve the problem for the scene at t = t_k, update the scene using the internal simulator
        and the derived ego policy and repeat until t_k = `horizon` or until the goal has been reached.

        This method changes the internal environment by forward simulating it over the prediction horizon.

        :param time_steps: how many time-steps shall be solved (not planning horizon !).
        :return: derived ego trajectory [horizon + 1, 5].
        :return: derived actual ado trajectories [num_ados, 1, horizon + 1, 5].
        """
        x5_opt = torch.zeros((time_steps + 1, 5))
        ado_traj = torch.zeros((self.env.num_ados, 1, time_steps + 1, 5))
        self.log_reset(log_horizon=time_steps)

        # Initialize trajectories with current state and environment time.
        x5_opt[0] = self._env.ego.state_with_time
        self.log_append(x5_planned=self.env.ego.unroll_trajectory(torch.zeros((self.T, 2)), dt=self.env.dt), tag="opt")
        for j, ghost in enumerate(self.env.ghosts):
            i_ado, i_mode = self.env.index_ghost_id(ghost_id=ghost.id)
            ado_traj[i_ado, i_mode, 0, :] = ghost.agent.state_with_time
        self.log_append(ado_planned=self.env.predict_wo_ego(t_horizon=self.T + 1), tag="opt")

        logging.info(f"Starting trajectory optimization solving for planning horizon {time_steps} steps ...")
        for k in range(time_steps):
            logging.info(f"solver {self.name} @k={k}: initializing optimization")
            self._iteration = k

            # Solve optimisation problem.
            ego_controls_k = self.determine_ego_controls(**solver_kwargs)
            assert check_ego_controls(ego_controls_k, t_horizon=self.T)
            logging.info(f"solver {self.name} @k={k}: finishing optimization")

            # For logging purposes unroll and predict the scene for the derived ego controls.
            x5_opt_planned = self.env.ego.unroll_trajectory(controls=ego_controls_k, dt=self.env.dt)
            self.log_append(x5_planned=x5_opt_planned, tag="opt")
            ado_planned = self._env.predict_w_controls(controls=ego_controls_k)
            self.log_append(ado_planned=ado_planned, tag="opt")

            # Forward simulate environment.
            ado_states, ego_state = self._eval_env.step(ego_control=ego_controls_k[0:1, :])
            self._env.step_reset(ego_state_next=ego_state, ado_states_next=ado_states)
            x5_opt[k + 1] = ego_state
            ado_traj[:, :, k + 1, :] = ado_states[:, :, 0, :]

            # If the goal state has been reached, break the optimization loop (and shorten trajectories to
            # contain only states up to now (i.e. k + 1 optimization steps instead of max_steps).
            if torch.norm(ego_state[0:2] - self._goal) < 0.1:
                x5_opt = x5_opt[:k + 1, :].detach()
                ado_traj = ado_traj[:, :, :k + 1, :].detach()
                break

            # Logging.
            self.intermediate_log()
            logging.info(f"solver {self.name} @k={k}: ego optimized controls = {ego_controls_k.tolist()}")
            logging.info(f"solver {self.name} @k={k}: ego optimized path = {x5_opt_planned[:, 0:2].tolist()}")

        logging.info(f"solver {self.name}: logging and visualizing trajectory optimization")
        self.env.detach()
        self.log_summarize()
        self.visualize_optimization()
        self.visualize_scenes()
        logging.info(f"solver {self.name}: finishing up optimization process")
        return x5_opt, ado_traj

    def determine_ego_controls(self, **solver_kwargs) -> torch.Tensor:
        """Determine the ego control inputs for the internally stated problem and the current state of the environment.
        The implementation crucially depends on the solver class itself and is hence not implemented here.

        :return: ego_controls: control inputs of ego agent for whole planning horizon.
        :return: optimization dictionary of solution (containing objective, infeasibility scores, etc.).
        """
        # Solve optimisation problem for each initial condition, either in multiprocessing or sequential.
        initial_values = zip(self.z0s_default(), self.cores)
        if self.do_multiprocessing:
            results = joblib.Parallel(n_jobs=8)(joblib.delayed(self.optimize)
                                                (z0, tag, **solver_kwargs) for z0, tag in initial_values)
            for i, (_, _, optimization_log) in enumerate(results):
                self._optimization_log.update({key: x for key, x in optimization_log.items() if self.cores[i] in key})
        else:
            results = [self.optimize(z0, tag, **solver_kwargs) for z0, tag in initial_values]

        # Return controls with minimal objective function result.
        index_best = int(np.argmin([obj for _, obj, _ in results]))
        z_opt_best, self._core_opt = results[index_best][0], self.cores[index_best]
        return self.z_to_ego_controls(z_opt_best.detach().numpy())

    @abstractmethod
    def optimize(self, z0: torch.Tensor, tag: str, **kwargs) -> Tuple[torch.Tensor, float, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    ###########################################################################
    # Initialization ##########################################################
    ###########################################################################
    @abstractmethod
    def initialize(self, **solver_params):
        raise NotImplementedError

    @abstractmethod
    def z0s_default(self, just_one: bool = False) -> torch.Tensor:
        raise NotImplementedError

    ###########################################################################
    # Problem formulation - Formulation #######################################
    ###########################################################################
    @abstractmethod
    def num_optimization_variables(self) -> int:
        raise NotImplementedError

    ###########################################################################
    # Problem formulation - Objective #########################################
    ###########################################################################
    @staticmethod
    def objective_defaults() -> List[Tuple[str, float]]:
        raise NotImplementedError

    def objective(self, z: np.ndarray, tag: str) -> float:
        x5 = self.z_to_ego_trajectory(z)
        objective = np.sum([m.objective(x5) for m in self._objective_modules.values()])

        logging.debug(f"solver {self.name}:{tag} Objective function = {objective}")
        ado_planned = torch.zeros(0)  # pseudo for ado_planned (only required for plotting)
        if self.verbose > 2:
            ado_planned = self.env.predict_w_trajectory(trajectory=x5)
        self.log_append(x5_planned=x5, obj_overall=objective, ado_planned=ado_planned, tag=tag)
        self.log_append(**{f"obj_{key}": mod.obj_current for key, mod in self._objective_modules.items()}, tag=tag)
        return float(objective)

    ###########################################################################
    # Problem formulation - Constraints #######################################
    ###########################################################################
    def optimization_variable_bounds(self) -> Tuple[List, List]:
        limits = self._env.ego.control_limits()
        lb = (np.ones(2 * self.num_optimization_variables()) * limits[0]).tolist()
        ub = (np.ones(2 * self.num_optimization_variables()) * limits[1]).tolist()
        return lb, ub

    @staticmethod
    def constraints_defaults() -> List[str]:
        raise NotImplementedError

    def constraints(self, z: np.ndarray, tag: str, return_violation: bool = False) -> np.ndarray:
        if self.is_unconstrained:
            return np.array([]) if not return_violation else (np.array([]), 0.0)

        x4 = self.z_to_ego_trajectory(z)
        constraints = np.concatenate([m.constraint(x4) for m in self._constraint_modules.values()])
        violation = float(np.sum([m.compute_violation() for m in self._constraint_modules.values()]))

        logging.debug(f"solver {self.name}:{tag}: Constraints vector = {constraints}")
        self.log_append(inf_overall=violation, tag=tag)
        self.log_append(**{f"inf_{key}": mod.inf_current for key, mod in self._constraint_modules.items()}, tag=tag)
        return constraints if not return_violation else (constraints, violation)

    @property
    def is_unconstrained(self) -> bool:
        return len(self._constraint_modules.keys()) == 0

    ###########################################################################
    # Utility #################################################################
    ###########################################################################
    @abstractmethod
    def z_to_ego_trajectory(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def z_to_ego_controls(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def _build_objective_modules(self, modules: List[Tuple[str, float]]) -> Dict[str, ObjectiveModule]:
        assert all([name in OBJECTIVES.keys() for name, _ in modules]), "invalid objective module detected"
        assert all([0.0 <= weight for _, weight in modules]), "invalid solver module weight detected"
        return {m: OBJECTIVES[m](horizon=self.T, weight=w, sim=self._env, goal=self.goal) for m, w in modules}

    def _build_constraint_modules(self, modules: List[str]) -> Dict[str, ConstraintModule]:
        assert all([name in CONSTRAINTS.keys() for name in modules]), "invalid constraint module detected"
        return {m: CONSTRAINTS[m](horizon=self.T, sim=self._env) for m in modules}

    ###########################################################################
    # Visualization & Logging #################################################
    ###########################################################################
    def intermediate_log(self):
        if self.verbose > 0 and self.optimization_log is not None:
            for tag in self.cores:
                k = self._iteration

                # Log first and last (considered as best) objective value.
                log = {key: self.optimization_log[f"{tag}/obj_{key}_{k}"] for key in self.objective_keys}
                log = {key: f"{log[key][0]:.4f} => {log[key][-1]:.4f}" for key in self.objective_keys}
                logging.info(f"solver [{tag}] - objectives: {log}")

                # Log first and last (considered as best) infeasibility value.
                log = {key: self.optimization_log[f"{tag}/inf_{key}_{k}"] for key in self.constraint_modules}
                log = {key: f"{log[key][0]:.4f} => {log[key][-1]:.4f}" for key in self.constraint_modules}
                logging.info(f"solver [{tag}] - infeasibility: {log}")

    @staticmethod
    def log_keys() -> List[str]:
        return ["x5_planned", "ado_planned"]

    def log_reset(self, log_horizon: int):
        # Reset iteration counter.
        self._iteration = 0

        if self.verbose > -1:
            self._optimization_log = {}
            for k in range(log_horizon):
                for tag in self.cores:
                    # Set default logging variables for cores.
                    self._optimization_log.update({f"{tag}/{key}_{k}": [] for key in self.log_keys()})
                    # Set logging variables for each objective and constraint module.
                    self._optimization_log.update({f"{tag}/obj_{key}_{k}": [] for key in self.objective_keys})
                    self._optimization_log.update({f"{tag}/inf_{key}_{k}": [] for key in self.constraint_keys})
                # Set default logging variables for opt.
                self._optimization_log.update({f"opt/{key}_{k}": [] for key in self.log_keys()})

    def log_append(self, tag: str, **kwargs):
        if self.verbose > -1 and self.optimization_log is not None:
            for key, value in kwargs.items():
                x = torch.tensor(value) if type(value) != torch.Tensor else value.detach()
                self._optimization_log[f"{tag}/{key}_{self._iteration}"].append(x)

    def log_summarize(self):
        """Summarize optimisation-step dictionaries to a single tensor per logging key, e.g. collapse all objective
        value tensors for k = 1, 2, ..., N to one tensor for the whole optimisation process.

        Attention: It is assumed that the last value of each series is the optimal value for this kth optimisation,
        e.g. the last value of the objective tensor `obj_overall` should be the smallest one. However it is hard to
        validate for the general logging key, therefore it is up to the user to implement it correctly.
        """
        if self.verbose > -1 and self.optimization_log is not None:
            objective_keys = [f"{tag}/obj_{key}" for key in self.objective_keys for tag in self.cores]
            constraint_keys = [f"{tag}/inf_{key}" for key in self.constraint_keys for tag in self.cores]
            log_tag_keys = [f"{tag}/{key}" for key in self.log_keys() for tag in self.cores]
            log_opt_keys = [f"opt/{key}" for key in self.log_keys()]

            for key in (objective_keys + constraint_keys + log_tag_keys + log_opt_keys):
                summary = [self.optimization_log[f"{key}_{k}"][-1] for k in range(self._iteration + 1)]
                self._optimization_log[key] = torch.stack(summary)

            # Restructure 1-size tensor to actual vectors (objective and constraint).
            for k in range(self._iteration):
                for key in (objective_keys + constraint_keys):
                    self._optimization_log[f"{key}_{k}"] = torch.stack(self._optimization_log[f"{key}_{k}"])

    def visualize_optimization(self):
        """Visualize optimization iterations by plotting the planned ego trajectory for every optimization step as
        well as the values of objective and infeasibility (constraint violation), iff verbose > 2."""
        if self.verbose > 2:
            from mantrap.evaluation.visualization import visualize
            assert self.optimization_log is not None
            output_directory_path = build_os_path(f"outputs/", make_dir=True, free=False)

            tags = np.unique([key.split("/")[0] for key in self.optimization_log.keys() if key.split("/")[0] != "opt"])
            for tag in tags:
                for k in range(self._iteration):
                    obj_dict, inf_dict = {}, {}
                    for key in self.objective_keys:
                        obj_dict[key] = self.optimization_log[f"{tag}/obj_{key}_{k}"]
                    for key in self.constraint_keys:
                        inf_dict[key] = self.optimization_log[f"{tag}/inf_{key}_{k}"]

                    ego_planned = torch.stack(self.optimization_log[f"{tag}/x5_planned_{k}"])
                    ado_planned = torch.stack(self.optimization_log[f"{tag}/ado_planned_{k}"])

                    visualize(ego_planned=ego_planned, ado_planned=ado_planned, ego_trials=None,
                              obj_dict=obj_dict, inf_dict=inf_dict, env=self.env, single_opt=True,
                              file_path=os.path.join(output_directory_path, f"{self.name}:{self.env.name}:{tag}_{k}"))

    def visualize_scenes(self):
        """Visualize planned trajectory over full time-horizon as well as simulated ado reactions (i.e. their
        trajectories conditioned on the planned ego trajectory), iff verbose > 1."""
        if self.verbose > 1:
            from mantrap.evaluation.visualization import visualize
            assert self.optimization_log is not None
            output_directory_path = build_os_path(f"outputs/", make_dir=True, free=False)

            # From optimization log extract the core (initial condition) which has resulted in the best objective
            # value in the end. Then, due to the structure demanded by the visualization function, repeat the entry
            # N=t_horizon times to be able to visualize the whole distribution at every time.
            obj_dict = {key: self.optimization_log[f"{self.core_opt}/obj_{key}"] for key in self.objective_keys}
            obj_dict = {key: [obj_dict[key]] * (self._iteration + 1) for key in self.objective_keys}
            inf_dict = {key: self.optimization_log[f"{self.core_opt}/inf_{key}"] for key in self.constraint_keys}
            inf_dict = {key: [inf_dict[key]] * (self._iteration + 1) for key in self.constraint_keys}

            x5_trials = [self._optimization_log[f"{self.core_opt}/x5_planned_{k}"] for k in range(self._iteration)]

            visualize(ego_planned=self.optimization_log["opt/x5_planned"],
                      ado_planned=self.optimization_log["opt/ado_planned"],
                      ego_trials=x5_trials, obj_dict=obj_dict, inf_dict=inf_dict, env=self.env,
                      file_path=os.path.join(output_directory_path, f"{self.name}:{self.env.name}:opt"))

    ###########################################################################
    # Solver parameters #######################################################
    ###########################################################################
    @property
    def env(self) -> GraphBasedEnvironment:
        return self._env

    @env.setter
    def env(self, env: GraphBasedEnvironment):
        self._env = env

    @property
    def eval_env(self) -> GraphBasedEnvironment:
        return self._eval_env

    @property
    def goal(self) -> torch.Tensor:
        return self._goal

    @property
    def T(self) -> int:
        return self._solver_params["t_planning"]

    @property
    def objective_modules(self) -> Dict[str, ObjectiveModule]:
        return self._objective_modules

    @property
    def objective_keys(self) -> List[str]:
        return ["overall"] + list(self.objective_modules.keys())

    @property
    def constraint_modules(self) -> Dict[str, ConstraintModule]:
        return self._constraint_modules

    @property
    def constraint_keys(self) -> List[str]:
        return ["overall"] + list(self.constraint_modules.keys())

    ###########################################################################
    # Utility parameters ######################################################
    ###########################################################################
    @property
    def cores(self) -> List[str]:
        return [f"core{i}" for i in range(self.z0s_default().shape[0])]

    @property
    def core_opt(self) -> str:
        return self._core_opt if self._core_opt is not None else "opt"

    @property
    def do_multiprocessing(self) -> bool:
        return self._solver_params["multiprocessing"]

    ###########################################################################
    # Logging parameters ######################################################
    ###########################################################################
    @property
    def optimization_log(self) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        return self._optimization_log

    @property
    def verbose(self) -> bool:
        return self._solver_params["verbose"]

    @property
    def solver_name(self) -> str:
        return self.__class__.__name__.lower()

    @property
    def name(self) -> str:
        return self.solver_name + "_" + self._config_name
