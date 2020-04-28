from typing import Dict, List, Tuple

import torch

from mantrap.constants import *
from mantrap.agents import IntegratorDTAgent
from mantrap.environment import ORCAEnvironment
from mantrap.solver.solver_intermediates.z_controls import ZControlIntermediate
from mantrap.utility.shaping import check_ego_action, check_ego_controls


class ORCASolver(ZControlIntermediate):
    """ORCA-based solver.

    Based on the ORCA algorithm defined in `ORCAEnvironment` this solver finds the optimal trajectory by
    introducing the ego agent as another "normal" agent in the scene, solving for the optimal velocity update
    for every agent in the scene (by following the ORCA algorithm and assumptions) and returning the velocity
    which is associated to the agent representing the ego agent.
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
        graph = orca_env.build_connected_graph_wo_ego(t_horizon=1, safe_time=self.planning_horizon * self.env.dt)
        ego_action = graph[f"{ID_EGO}_0_0_{GK_CONTROL}"]  # first "0" because first (and only) mode !
        assert check_ego_action(ego_action)

        # ORCA itself only plans for one step ahead, but can guarantee some minimal collision-free time interval,
        # given all other agents behave according to the ORCA algorithm (which is not the case but the underlying
        # assumption here), therefore stretch the control action over the full planning horizon.
        ego_controls = torch.stack([ego_action] * self.planning_horizon)
        assert check_ego_controls(ego_controls, t_horizon=self.planning_horizon)

        # Determine z and objective value from determined ego-control. For logging only also determine
        # the constraints values (since logging happens within the function).
        z = ego_controls.flatten()
        objective_value = self.objective(z=z.detach().numpy(), tag=tag, ado_ids=ado_ids)
        _ = self.constraints(z=z.detach().numpy(), tag=tag, ado_ids=ado_ids)
        return z, objective_value, self.log

    ###########################################################################
    # Initialization ##########################################################
    ###########################################################################
    def initialize(self, **solver_params):
        assert self.env.ego.__class__ == IntegratorDTAgent

    def initial_values(self, just_one: bool = False) -> torch.Tensor:
        """As explained in `_optimize()` the optimization is independent from the initial value of z, therefore
        only return one value, to enforce single-threaded computations. """
        z0 = torch.zeros((1, self.planning_horizon, 2))
        return z0 if not just_one else z0[0, :, :]

    ###########################################################################
    # Problem formulation - Formulation #######################################
    ###########################################################################
    @staticmethod
    def objective_defaults() -> List[Tuple[str, float]]:
        return [(OBJECTIVE_GOAL, 1.0)]

    @staticmethod
    def constraints_defaults() -> List[str]:
        return [CONSTRAINT_CONTROL_LIMIT]

    ###########################################################################
    # Solver properties #######################################################
    ###########################################################################
    @property
    def solver_name(self) -> str:
        return "orca"
