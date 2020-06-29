import math
import os
import sys
import typing

import torch

import mantrap.constants

from ..base import TrajOptSolver


class RRTStarSolver(TrajOptSolver):

    def optimize_core(
        self,
        z0: torch.Tensor,
        ado_ids: typing.List[str],
        tag: str = mantrap.constants.TAG_OPTIMIZATION,
        **solver_kwargs
    ) -> typing.Tuple[torch.Tensor, typing.Dict[str, torch.Tensor]]:
        """Optimization function for single core to find optimal z-vector.

        Given some initial value `z0` find the optimal allocation for z with respect to the internally defined
        objectives and constraints. This function is executed in every thread in parallel, for different initial
        values `z0`. To simplify optimization not all agents in the scene have to be taken into account during
        the optimization but only the ones with ids defined in `ado_ids`.

        Implementation of optimal sampling-based path planning algorithm RRT* by Karaman and Frazzoli
        (Sampling-based Algorithms for Optimal Motion Planning). Since the algorithm is (runtime) asymptotically
        optimal and complete, search until the runtime has ended, then convert the path into robot controls,
        The implementation from PythonRobotics (https://github.com/AtsushiSakai/PythonRobotics) is used here.

        This solver only takes into account the current states, not any future state, which makes it efficient
        to solve, but also not very much anticipative.

        :param z0: initial value of optimization variables.
        :param tag: name of optimization call (name of the core).
        :param ado_ids: identifiers of ados that should be taken into account during optimization.
        :returns: z_opt (optimal values of optimization variable vector)
                  optimization_log (logging dictionary for this optimization = self.log)
        """
        self.import_modules()
        from rrt_star import RRTStar
        from rrt_utils import path_smoothing

        # Get current environment state and boundaries as well as goal.
        ego_state, ado_states = self.env.states()
        start = ego_state[0:2].detach().numpy().tolist()
        goal = self.goal.detach().numpy().tolist()
        bounds = tuple(self.env.x_axis)
        assert self.env.x_axis == self.env.y_axis  # assumption by PythonRobotics implementation

        # Create list of circular obstacles around pedestrians as nested list.
        obstacles = []
        ado_radius = mantrap.constants.RRT_PED_RADIUS
        for i, ado_id in enumerate(ado_ids):
            m_ado = self.env.index_ado_id(ado_id=ado_id)
            ado_position = ado_states[m_ado, 0:2].detach().numpy().tolist()
            obstacles.append([ado_position[0], ado_position[1], ado_radius])

        # Initialize and execute RRTStar algorithm.
        rrt = RRTStar(start=start, goal=goal, rand_area=bounds, obstacle_list=obstacles,
                      max_iter=mantrap.constants.RRT_ITERATIONS,
                      connect_circle_dist=mantrap.constants.RRT_REWIRE_RADIUS,
                      expand_dis=1.0, path_resolution=0.1,
                      goal_sample_rate=mantrap.constants.RRT_GOAL_SAMPLING_PROBABILITY)
        path = rrt.planning(search_until_max_iter=True, animation=False)

        # Smooth resulting path and convert into custom format.
        if path is None:
            path = [start, goal]
        else:
            path = path_smoothing(path=path, max_iter=1000, obstacle_list=obstacles)
            path.reverse()  # returned list from goal -> start, so reverse it
            path = [path[i] for i in range(len(path)) if i == 0 or
                    math.hypot(path[i-1][0] - path[i][0], path[i-1][1] - path[i][1]) > 0.01]  # remove duplicates

        # Extract ego controls by converting the path to a trajectory, and transform the trajectory to
        # controls using the ego's internal dynamics.
        ego_trajectory = self.env.ego.expand_trajectory(torch.tensor(path), dt=self.env.dt)
        ego_controls = self.env.ego.roll_trajectory(ego_trajectory, dt=self.env.dt)
        ego_controls = self.env.ego.make_controls_feasible(controls=ego_controls)
        if ego_controls.shape[0] < self.planning_horizon:
            len_diff = self.planning_horizon - ego_controls.shape[0]
            ego_controls = torch.cat((ego_controls, torch.zeros((len_diff, 2))), dim=0)
        return ego_controls[:self.planning_horizon, :], self.logger.log

    ###########################################################################
    # RRT Helpers #############################################################
    ###########################################################################
    @staticmethod
    def import_modules():
        if RRTStarSolver.module_os_path() not in sys.path:
            sys.path.insert(0, RRTStarSolver.module_os_path())

    @staticmethod
    def module_os_path() -> str:
        module_path = mantrap.utility.io.build_os_path("third_party/RRT", make_dir=False, free=False)
        assert os.path.isdir(module_path)
        return module_path

    ###########################################################################
    # Solver properties #######################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "rrt"
