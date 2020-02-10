from collections import defaultdict
import logging
import os
from typing import List, Tuple, Union

import ipopt
import numpy as np
import torch

from mantrap.constants import agent_speed_max, igrad_max_solver_steps, igrad_max_solver_cpu_time
from mantrap.simulation.simulation import GraphBasedSimulation
from mantrap.solver.solver import Solver
from mantrap.utility.io import path_from_home_directory
from mantrap.utility.maths import Derivative2
from mantrap.utility.shaping import check_trajectory_primitives


class IGradSolver(Solver):
    def __init__(self, sim: GraphBasedSimulation, goal: torch.Tensor, **solver_params):
        super(IGradSolver, self).__init__(sim, goal, **solver_params)

        # Prepare solver for solving by pre-computing solver state independent variables.
        self._ado_states_wo_np = self._env.predict(self.T, ego_trajectory=None).detach().numpy()
        self._goal_np = self.goal.detach().numpy()
        self._derivative_2 = Derivative2(horizon=self.T, dt=self._env.dt, num_axes=4)
        self._ado_states_wo_ddt_np = self._derivative_2.compute(self._ado_states_wo_np)

        # Logging variables. Using default-dict(list) whenever a new entry is created, it does not have to be checked
        # whether the related key is already existing, since if it is not existing, it is created with a list as
        # starting value, to which the new entry is appended.
        self._optimization_log = defaultdict(list) if self.is_verbose else None
        self._objective_dict_latest = None
        self._grad_latest = None
        self._x_latest = None  # iteration() function does not input x (!)

    def _determine_ego_action(self, env: GraphBasedSimulation) -> torch.Tensor:
        raise NotImplementedError

    def _solve_optimization(
        self,
        x0: torch.Tensor,
        max_iter: int = igrad_max_solver_steps,
        max_cpu_time: float = igrad_max_solver_cpu_time,
        approx_jacobian: bool = False,
        approx_hessian: bool = True,
        check_derivative: bool = False,
    ):
        """Solve optimization problem by finding constraint bounds, constructing ipopt optimization problem and
        solve it using the parameters defined in the function header."""
        assert check_trajectory_primitives(x0, t_horizon=self.T), "x0 should be ego trajectory"

        # Formulate optimization problem as in standardized IPOPT format.
        x_init, x0_flat = x0[0, :].detach().numpy(), x0.flatten().numpy()
        lb, ub, cl, cu = self.constraint_bounds(x_init=x_init)

        nlp = ipopt.problem(n=2 * self.T, m=len(cl), problem_obj=self, lb=lb, ub=ub, cl=cl, cu=cu)
        nlp.addOption("max_iter", max_iter)
        nlp.addOption("max_cpu_time", max_cpu_time)
        if approx_jacobian:
            nlp.addOption("jacobian_approximation", "finite-difference-values")
        if approx_hessian:
            nlp.addOption("hessian_approximation", "limited-memory")
        nlp.addOption("print_level", 5)  # the larger the value, the more print output.
        if self.is_verbose or check_derivative:
            nlp.addOption("print_timing_statistics", "yes")
            nlp.addOption("derivative_test", "first-order")
            nlp.addOption("derivative_test_tol", 1e-4)

        # Solve optimization problem for "optimal" ego trajectory `x_optimized`.
        x_optimized, info = nlp.solve(x0_flat)
        x_optimized = np.reshape(x_optimized, (self.T, 2))

        # Plot optimization progress.
        self.log_and_clean_up()

        return torch.from_numpy(x_optimized)

    ###########################################################################
    # Optimization formulation ################################################
    # IPOPT requires to use numpy arrays for computation, therefore switch ####
    # everything from torch to numpy here #####################################
    ###########################################################################

    def objective(self, x: np.ndarray) -> float:
        """The objective is to minimize the interaction between the ego and ado, which can be expressed as the
        the L2 difference between the position of every agent with respect to interaction with ego and without
        taking it into account.

        J(x_{ego}) = sum_{t = 0}^T sum_{m = 0}^M || x_m^t(x_{ego}^t) - x_m_{wo}^t
        """
        assert self._env.num_ado_modes == 1, "currently only uni-modal agents are supported"
        x2 = torch.from_numpy(x).view(self.T, 2)
        assert check_trajectory_primitives(x2, t_horizon=self.T), f"x should be ego trajectory with length {self.T}"

        objective_dict = dict()
        # interaction objective (as least interfering as possible).
        objective_dict["obj_interaction"] = self._objective_interaction(x2)
        # goal distance for every point (as fast as possible).
        objective_dict["obj_goal"] = self._objective_goal(x2)
        objective = np.sum(list(objective_dict.values()))

        logging.debug(f"Objective function = {objective}")
        if self.is_verbose:
            self._objective_dict_latest = objective_dict
            self._x_latest = x.copy()  # logging most current optimization values
        return float(objective)

    # def _objective_interaction(self, x2: torch.Tensor) -> float:
    #     ado_states_np = self._env.predict(self.T, ego_trajectory=x2).detach().numpy()
    #     return np.linalg.norm(ado_states_np[:, :, :, 0:2] - self._ado_states_wo_np[:, :, :, 0:2], axis=3).sum()

    def _objective_interaction(self, x2: torch.Tensor) -> float:
        ado_states_np = self._env.predict(self.T, ego_trajectory=x2).detach().numpy()
        ado_states_ddt_np = self._derivative_2.compute(ado_states_np[:, :, :, 0:2])
        return np.linalg.norm(ado_states_ddt_np - self._ado_states_wo_ddt_np[:, :, :, 0:2], axis=3).sum()

    def _objective_goal(self, x2: torch.Tensor) -> float:
        goal_distances = np.linalg.norm(x2 - self._goal_np, axis=1)
        weights = np.linspace(0, 1, num=x2.shape[0])
        return goal_distances.dot(weights)

    ################
    # Gradient #####
    ################

    def gradient(self, x: np.ndarray) -> np.ndarray:
        assert self._env.num_ado_modes == 1, "currently only uni-modal agents are supported"
        x2 = torch.from_numpy(x).view(self.T, 2)
        assert check_trajectory_primitives(x2, t_horizon=self.T), f"x should be ego trajectory with length {self.T}"

        gradient = np.zeros(2 * self.T)

        # Interaction objective gradient.
        gradient += self._gradient_interaction(x2)
        # Goal distance gradient.
        gradient += self._gradient_goal(x2)

        logging.debug(f"Gradient function = {gradient}")
        if self.is_verbose:
            self._x_latest = x.copy()  # logging most current optimization values
            self._grad_latest = gradient.copy()
        return gradient

    def _gradient_interaction(self, x2: torch.Tensor) -> np.ndarray:
        gradient = np.zeros(2 * self.T)

        # Predict the ado states for the next `self.T` time-steps (although the full state is predicted here, we
        # merely need the agent's future positions going further.
        ado_states_np = self._env.predict(self.T, ego_trajectory=x2).detach().numpy()

        # Compute partial gradient using simulation graph derivative, expressing the correlation between the movement
        # of an ado and the ego's trajectory.
        graphs = self._env.build_connected_graph(ego_positions=x2)
        ego_positions = [graphs[f"ego_{k}_position"] for k in range(self.T)]
        partial_grads = torch.zeros((self.T, self.T, self.M, 2))
        for k in range(self.T):
            for m in range(self.M):
                ado_output = graphs[f"{self._env.ado_ghosts[m].gid}_{k}_output"]
                grads_tuple = torch.autograd.grad(ado_output, inputs=ego_positions[:k + 1], retain_graph=True)
                partial_grads[:, :k + 1, m, :] = torch.stack(grads_tuple)
        partial_grads_np = partial_grads.detach().numpy()

        # Compute gradient using analytically derived formula.
        # diff = ado_states_np[:, :, :, 0:2] - self._ado_states_wo_np[:, :, :, 0:2]
        # norm = np.linalg.norm(diff, axis=3)
        # norm[norm < 1e-6] = np.inf  # if norm is zero, i.e. equivalent ddts, then gradient in this direction is 0 too
        # for k in range(self.T):
        #     for t in range(k, self.T):
        #         partials_x = partial_grads_np[k, t, :, 0]
        #         partials_y = partial_grads_np[k, t, :, 1]
        #         gradient[2 * k] += np.sum(1 / norm[:, :, t] * np.sum(diff[:, :, t, 0], axis=1) * partials_x)
        #         gradient[2 * k + 1] += np.sum(1 / norm[:, :, t] * np.sum(diff[:, :, t, 1], axis=1) * partials_y)

        ado_states_ddt_np = self._derivative_2.compute(ado_states_np[:, :, :, 0:2])
        diff = ado_states_ddt_np - self._ado_states_wo_ddt_np[:, :, :, 0:2]
        norm = np.linalg.norm(diff, axis=3)
        norm[norm < 1e-6] = np.inf  # if norm is zero, i.e. equivalent ddts, then gradient in this direction is 0 too
        for k in range(1, self.T):
            for t in range(k - 1, self.T - 1):
                partials_x = partial_grads_np[k, t - 1, :, 0] - 2 * partial_grads_np[k, t, :, 0]
                partials_y = partial_grads_np[k, t - 1, :, 1] - 2 * partial_grads_np[k, t, :, 1]
                gradient[2 * k] += np.sum(1 / norm[:, :, t] * np.sum(diff[:, :, t, 0], axis=1) * partials_x)
                gradient[2 * k + 1] += np.sum(1 / norm[:, :, t] * np.sum(diff[:, :, t, 1], axis=1) * partials_y)

        return gradient

    def _gradient_goal(self, x2: torch.Tensor) -> np.ndarray:
        gradient = np.zeros(2 * self.T)

        weights = np.linspace(0, 1, num=x2.shape[0])
        diff_goal = x2 - self._goal_np
        norm_goal = np.linalg.norm(diff_goal, axis=1)
        for k in range(self.T):
            if norm_goal[k] < 1e-3:
                continue
            gradient[2 * k] += 1 / norm_goal[k] * diff_goal[k, 0] * weights[k]
            gradient[2 * k + 1] += 1 / norm_goal[k] * diff_goal[k, 1] * weights[k]

        return gradient

    ################
    # Constraints ##
    ################

    def constraints(self, x: np.ndarray) -> np.ndarray:
        """An unconstrained optimisation would result in pushing the ego as far as possible from the agents,
        ensuring minimal interaction but also leading to infinite (or at least very large) control effort on the
        one side and not reaching the goal in a reasonable amount of time on the other side. Therefore the trajectory
        length of the ego trajectory is bounded. The most exact but also most computationally expensive way doing
        that is by constraining subsequent trajectory points

        || x_{ego}^t - x_{ego}^{t - 1} || < gamma * dt

        for some maximal velocity gamma and the (simulation) time-step dt. However this would introduce T - 1
        quadratic (at least convex) constraints. Therefore another option would be to set a bound by constraining the
        sum over the whole trajectory only:

        sum_{t = 1}^T || x_{ego}^t - x_{ego}^{t - 1} || < gamma * T
        """
        x2 = np.reshape(x, (self.T, 2))
        inter_path_distance = np.linalg.norm(x2[1:, :] - x2[:-1, :], axis=1)
        initial_position = x2[0, :]

        constraints = np.hstack((inter_path_distance, initial_position))
        logging.debug(f"Constraints vector = {constraints}")
        if self.is_verbose:
            self._x_latest = x.copy()  # logging most current optimization values
        return constraints

    def constraint_bounds(
        self, x_init: np.ndarray
    ) -> Tuple[List[Union[None, float]], List[Union[None, float]], List[Union[None, float]], List[Union[None, float]]]:
        assert x_init.size == 2, "initial position should be two-dimensional"

        # External constraint bounds (inter-point distance and initial point equality).
        cl = [None] * (self.T - 1) + [x_init[0], x_init[1]]
        cu = [agent_speed_max * self._env.dt] * (self.T - 1) + [x_init[0], x_init[1]]

        # Optimization variable bounds.
        lb = (np.ones(2 * self.T) * self._env.axes[0][0]).tolist()
        ub = (np.ones(2 * self.T) * self._env.axes[0][1]).tolist()

        return lb, ub, cl, cu

    ################
    # Jacobian #####
    ################

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        jacobian = np.zeros((2 + (self.T - 1)) * 2 * self.T)  # (2 + self.T - 1) constraints, derivative each wrt to x_i
        x2 = np.reshape(x, (self.T, 2))
        diff = x2[1:] - x2[:-1]
        norm = np.linalg.norm(diff, axis=1) + 1e-6  # prevent zero division

        # inter-point distance constraint jacobian - x and y coordinate.
        for i in range(self.T - 1):
            jacobian[i * 2 * self.T + 2 * i] = -1 / norm[i] * diff[i, 0]
            jacobian[i * 2 * self.T + 2 * i + 1] = -1 / norm[i] * diff[i, 1]
            jacobian[i * 2 * self.T + 2 * (i + 1)] = 1 / norm[i] * diff[i, 0]
            jacobian[i * 2 * self.T + 2 * (i + 1) + 1] = 1 / norm[i] * diff[i, 1]

        # initial position constraint jacobian.
        jacobian[(self.T - 1) * 2 * self.T] = 1
        jacobian[self.T * 2 * self.T + 1] = 1

        logging.debug(f"Constraint jacobian function computed")
        if self.is_verbose:
            self._x_latest = x.copy()  # logging most current optimization values
        return jacobian

    ################
    # Hessian ######
    ################
    # wrong hessian should just affect rate of convergence, not convergence in general
    # (given it is semi-positive definite which is the case for the identity matrix)
    # hessian = np.eye(3*self.O)

    def hessian(self, x, lagrange=None, obj_factor=None) -> np.ndarray:
        raise NotImplementedError

    ################
    # Utility ######
    ################

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, *args):
        if self.is_verbose:
            self._optimization_log["iter_count"] += [iter_count]
            self._optimization_log["obj_value"] += [obj_value]
            if self._objective_dict_latest is not None:
                for key, value in self._objective_dict_latest.items():
                    self._optimization_log[key] += [value]
            self._optimization_log["inf_pr"] += [inf_pr]
            self._optimization_log["d_norm"] += [d_norm]
            self._optimization_log["x"] += [self._x_latest]
            self._optimization_log["grad"] += [self._grad_latest]

    @property
    def T(self) -> int:
        return self.planning_horizon

    @property
    def M(self) -> int:
        return self._env.num_ados

    ###########################################################################
    # Visualization ###########################################################
    ###########################################################################

    def log_and_clean_up(self):
        """Clean up optimization logs and reset optimization parameters.
        IPOPT determines the CPU time including the intermediate function, therefore if we would plot at every step,
        we would loose valuable optimization time. Therefore the optimization progress is plotted all at once at the
        end of the optimization process."""

        # Plotting only makes sense if you see some progress in the optimization, i.e. compare and figure out what
        # the current optimization step has changed.
        if not self.is_verbose or len(self._optimization_log["iter_count"]) < 2:
            return

        # Find path to output directory, create it or delete every file inside.
        output_directory_path = os.path.join(path_from_home_directory("test/graphs/igrad_optimization", make_dir=True))
        file_list = [f for f in os.listdir(output_directory_path) if f.endswith(".png")]
        for f in file_list:
            os.remove(os.path.join(output_directory_path, f))

        # Plot optimization history.
        import matplotlib.pyplot as plt

        x2_base_np = np.reshape(self._optimization_log["x"][0], (self.T, 2))
        ado_traj_base_np = self._env.predict(self.T, ego_trajectory=torch.from_numpy(x2_base_np)).detach().numpy()
        for k in range(1, self._optimization_log["iter_count"][-1]):
            x2_np = np.reshape(self._optimization_log["x"][k], (self.T, 2))
            ado_traj_np = self._env.predict(self.T, ego_trajectory=torch.from_numpy(x2_np)).detach().numpy()

            fig = plt.figure(figsize=(15, 15), constrained_layout=True)
            plt.title(f"iGrad optimization - IPOPT step {k}")
            plt.axis("off")
            vis_keys = ["obj_value", "inf_pr", "d_norm"]
            if self._objective_dict_latest is not None:
                vis_keys += self._objective_dict_latest.keys()
            grid = plt.GridSpec(len(vis_keys) + 1, len(vis_keys), wspace=0.4, hspace=0.3)

            # Plot current and base solution in the scene. This includes the determined ego trajectory (x) as well as
            # the resulting ado trajectories based on some simulation.
            ax = fig.add_subplot(grid[: len(vis_keys), :])
            ax.plot(x2_np[:, 0], x2_np[:, 1], label="ego_current")
            ax.plot(x2_base_np[:, 0], x2_base_np[:, 1], label="ego_base")
            # Plot current and base resulting simulated ado trajectories in the scene.
            for m in range(self._env.num_ados):
                for g in range(self._env.num_ado_modes):
                    ax.plot(ado_traj_np[m, g, :, 0], ado_traj_np[m, g, :, 1], "--", label=f"ado_current_{m}")
                    ax.plot(ado_traj_base_np[m, g, :, 0], ado_traj_base_np[m, g, :, 1], "--", label=f"ado_base_{m}")
            ax.set_xlim(self.env.axes[0])
            ax.set_ylim(self.env.axes[1])
            plt.grid()
            plt.legend()

            # Plot several parameter describing the optimization process.
            for i, key in enumerate(vis_keys):
                ax = fig.add_subplot(grid[len(vis_keys), i])
                vis_data = np.log(np.asarray(self._optimization_log[key][:k]) + 1e-8)
                ax.plot(self._optimization_log["iter_count"][:k], vis_data)
                ax.set_title(f"log_{key}")
                plt.grid()

            plt.savefig(os.path.join(output_directory_path, f"{k}.png"), dpi=60)
            plt.close()

        # Reset optimization logging parameters for next optimization.
        self._optimization_log = defaultdict(list) if self.is_verbose else None
        self._x_latest = self._grad_latest = None
