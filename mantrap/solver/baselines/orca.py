from typing import Dict, List, Tuple

import numpy as np
import torch

from mantrap.constants import *
from mantrap.agents import IntegratorDTAgent
from mantrap.environment import ORCAEnvironment
from mantrap.solver.solver import Solver
from mantrap.utility.shaping import check_ego_action, check_ego_controls


class ORCASolver(Solver):
    """Implementation of baseline ego strategy according to 'Reciprocal n-body Collision Avoidance' by Jur van den Berg,
    Stephen J. Guy, Ming Lin, and Dinesh Manocha (short: ORCA, O = optimal). This strategy determines the optimal
    velocity update so that a) there is guaranteed no collision in the next time-steps (`agent_safe_dt`) and b) the
    new velocity is the closest velocity to the preferred velocity (here: constant velocity to goal) there is in the
    set of non-collision velocities. Thereby we have some main assumptions:

    1) all agents behave according to the ORCA formalism
    2) all agents are single integrators i.e. do not have holonomic constraints
    3) synchronized discrete time updates
    4) perfect observability of every agents state at the current time (pref. velocities unknown)

    In order to find the "optimal" velocity linear constraints are derived using the ORCA formalism and solved in
    a linear program. Since here only the ego behaves according to ORCA, every collision avoidance movement is fully
    executed by the ego itself, instead of splitting it up between the two avoiding agents, as in the original ORCA
    implementation.
    """

    def _optimize(self, z0: torch.Tensor, tag: str, ado_ids: List[str], **kwargs
                  ) -> Tuple[torch.Tensor, float, Dict[str, torch.Tensor]]:
        """Optimization function for single core to find optimal z-vector.

         Given some initial value `z0` find the optimal allocation for z with respect to the internally defined
         objectives and constraints. This function is executed in every thread in parallel, for different initial
         values `z0`. To simplify optimization not all agents in the scene have to be taken into account during
         the optimization but only the ones with ids defined in `ado_ids`.

         ORCA assumes that all agents in the scene behave as ORCA-Agents, i.e. are steerable using the ORCA
         algorithm. Then a collision-free control input for the ego agent can be computed. To compute the
         robot's control inputs the environment is re-initialized as `ORCAEnvironment` (in which all agents
         have like orca-like), with all agents in the scene including the robot itself. Then the robot is
         forward-predicted for one step, the controls for the ego-agent are extracted and returned.
         Notice that this approach is fully independent from the initial value of z (=z0) since the controls
         are defined fully based on the current (known) state of every agent.

         :param z0: initial value of optimization variables.
         :param tag: name of optimization call (name of the core).
         :param ado_ids: identifiers of ados that should be taken into account during optimization.
         :returns: z_opt (optimal values of optimization variable vector)
                   objective_opt (optimal objective value)
                   optimization_log (logging dictionary for this optimization = self.log)
         """
        orca_env = self.env.copy(env_type=ORCAEnvironment)

        # In order to be able to predict the next state for the ego agent without previously knowing it and
        # using the environment interface, another ado agent is added to the scene, being equal to the ego.
        # This enables predicting the scene without ego later on.
        orca_env.add_ado(
            position=orca_env.ego.position,
            velocity=orca_env.ego.velocity,
            goal=self.goal,
            num_modes=1,
            identifier=ID_EGO
        )
        graph = orca_env.build_connected_graph_wo_ego(t_horizon=1, safe_time=self.T*self.env.dt)
        ego_action = graph[f"{ID_EGO}_0_0_{GK_CONTROL}"]  # first "0" because first (and only) mode !
        assert check_ego_action(ego_action)

        # ORCA itself only plans for one step ahead, but can guarantee some minimal collision-free time interval,
        # given all other agents behave according to the ORCA algorithm (which is not the case but the underlying
        # assumption here), therefore stretch the control action over the full planning horizon.
        ego_controls = torch.stack([ego_action] * self.T)
        assert check_ego_controls(ego_controls, t_horizon=self.T)

        # Determine z and objective value from determined ego-control.
        z = ego_controls.flatten()
        objective_value = self.objective(z=z.detach().numpy())
        return z, objective_value, self.log

    ###########################################################################
    # Initialization ##########################################################
    ###########################################################################
    def initialize(self, **solver_params):
        assert self.env.ego.__class__ == IntegratorDTAgent

    def z0s_default(self, just_one: bool = False) -> torch.Tensor:
        """As explained in `_optimize()` the optimization is independent from the initial value of z, therefore
        only return one value, to enforce single-threaded computations. """
        return torch.zeros((1, 2))

    ###########################################################################
    # Problem formulation - Formulation #######################################
    ###########################################################################
    def num_optimization_variables(self) -> int:
        return 2  # [vx, vy]_ego

    @staticmethod
    def objective_defaults() -> List[Tuple[str, float]]:
        return [(OBJECTIVE_GOAL, 1.0)]

    @staticmethod
    def constraints_defaults() -> List[str]:
        return [CONSTRAINT_MAX_SPEED, CONSTRAINT_MIN_DISTANCE]

    ###########################################################################
    # Utility #################################################################
    ###########################################################################
    def z_to_ego_trajectory(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        controls = self.z_to_ego_controls(z, return_leaf=return_leaf)
        return self.env.ego.unroll_trajectory(controls, dt=self.env.dt)

    def z_to_ego_controls(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        return torch.from_numpy(z).view(-1, 2)
