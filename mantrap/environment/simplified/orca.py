from collections import namedtuple
from typing import Dict, List, Tuple, Union

import torch

from mantrap.agents.agent import Agent
from mantrap.agents import IntegratorDTAgent
from mantrap.constants import *
from mantrap.environment.environment import GraphBasedEnvironment
from mantrap.environment.iterative import IterativeEnvironment
from mantrap.utility.shaping import check_goal


class ORCAEnvironment(IterativeEnvironment):
    """ORCA-based environment.

    Implementation enviornment according to 'Reciprocal n-body Collision Avoidance' by Jur van den Berg,
    Stephen J. Guy, Ming Lin, and Dinesh Manocha (short: ORCA, O = optimal). In this environment the optimal velocity
    update is determined for every agent so that a) there is guaranteed no collision in the next time-steps
    (`agent_safe_dt`) and b) the new velocity is the closest velocity to the preferred velocity (here: constant
    velocity to goal) there is in the set of non-collision velocities. Thereby we have some main assumptions:

    1) all agents behave according to the ORCA formalism
    2) all agents are single integrators i.e. do not have holonomic constraints
    3) synchronized discrete time updates
    4) perfect observability of every agents state at the current time (pref. velocities unknown)

    In order to find the "optimal" velocity linear constraints are derived using the ORCA formalism and solved in
    a linear program. The ego agent is then regarded as one of the (N+1) agents in the scene and the scene, just
    with a known control input. Since ORCA assumes all parameters to be shared and known to the other agents it
    can neither be probabilistic nor multi-modal.

    Also the update of some agent is affected by the ego, if and only if the ego agent imposes an active constraint
    on this agent, which is usually not the case for every agent. Therefore when differentiating the ado positions
    with respect to the ego's state trajectory, usually it will no gradient (i.e. no connection in the computation
    graph), while in case of a connection having a very long network connection (i.e. comparably large
    computational effort to compute gradient).
    """

    LineConstraint = namedtuple("LineConstraint", ["point", "direction"])

    ###########################################################################
    # Scene ###################################################################
    ###########################################################################
    def add_ado(self, goal: torch.Tensor = torch.zeros(2), **ado_kwargs) -> Agent:
        assert check_goal(goal)
        params = [{PK_GOAL: goal.detach().double()}]
        return super(ORCAEnvironment, self).add_ado(IntegratorDTAgent, arg_list=params, **ado_kwargs)

    ###########################################################################
    # Simulation Graph ########################################################
    ###########################################################################
    def build_graph(
        self,
        ego_state: torch.Tensor = None,
        k: int = 0,
        safe_time: float = ORCA_SAFE_TIME,
        **graph_kwargs
    ) -> Dict[str, torch.Tensor]:
        # Graph initialization - Add ados and ego to graph (position, velocity and goals).
        graph_k = self.write_state_to_graph(ego_state, k=k, **graph_kwargs)

        # ORCA environment loop. In order to function properly for ORCA the re-planning time-interval (= dt) has
        # to be quite low, < 0.1 s. Since the simulation runs quite fast, we can split the actual simulation
        # time-step in smaller sub-steps. This improves the result while being independent from the user-defined
        # simulation time-step, at a comparably low computational cost.
        assert (self.dt / ORCA_SUB_TIME_STEP).is_integer()
        graph_kk = graph_k.copy()
        for _ in range(int(self.dt // ORCA_SUB_TIME_STEP)):
            for m_ghost, ghost in enumerate(self.ghosts):
                # Find set of line constraint for (current iteration) agent to be safe. Important here is that
                # uni-modality is assumed (and enforced during agent initialization). Therefore no inter-mode
                # interactions has to be taken into account, since they are not defined for ORCA anyway.
                others_ids = self.ghost_ids[:m_ghost] + self.ghost_ids[m_ghost + 1:]
                if ego_state is not None:
                    others_ids += [self.ego.id]
                constraints = self.line_constraints(
                    graph=graph_kk,
                    k=k,
                    self_id=ghost.id,
                    other_ids=others_ids,
                    dt=ORCA_SUB_TIME_STEP,
                    agent_radius=ORCA_AGENT_RADIUS,
                    agent_safe_dt=safe_time
                )

                # Find new velocity based on line constraints. Preferred velocity would be the going from the
                # current position to the goal directly with maximal speed. However when the goal has been reached
                # set the preferred velocity to zero (not original ORCA).
                speed_max = ghost.agent.speed_max
                vel_preferred = ghost.params[PK_GOAL] - ghost.agent.position
                goal_distance = torch.norm(vel_preferred)
                if goal_distance.item() < ORCA_MAX_GOAL_DISTANCE:
                    vel_preferred = torch.zeros(2)
                else:
                    vel_preferred = vel_preferred / goal_distance * speed_max

                # Solve the constrained optimization problem defined above.
                i_fail, vel_new = self.linear_solver_2d(constraints, speed_max=speed_max, velocity_opt=vel_preferred)
                if i_fail < len(constraints):
                    vel_new = self.linear_solver_3d(constraints, i_fail, speed_max=speed_max, velocity_new=vel_new)

                # Update graph based on findings for current agent. The graph variables are updated in every sub-
                # time-step, creating a differentiable chain since the next sub-time-step is then using these
                # updated values.
                trajectory = ghost.agent.unroll_trajectory(controls=vel_new.unsqueeze(dim=0),
                                                           dt=ORCA_SUB_TIME_STEP)
                graph_kk[f"{ghost.id}_{k}_{GK_POSITION}"] = trajectory[-1, 0:2]
                graph_kk[f"{ghost.id}_{k}_{GK_VELOCITY}"] = trajectory[-1, 2:4]
                graph_kk[f"{ghost.id}_{k}_{GK_CONTROL}"] = vel_new  # single integrator !

        # The result of all the computation above is to compute the control input for every ado in the
        # scene conditioned on the ego movement. The other graph-dict-entries has been written into
        # the graph during initialization as they represent the current internal state.
        for ghost_id in self.ghost_ids:
            graph_k[f"{ghost_id}_{k}_{GK_CONTROL}"] = graph_kk[f"{ghost_id}_{k}_{GK_CONTROL}"]

        return graph_k

    ###########################################################################
    # ORCA specific solver methods ############################################
    ###########################################################################
    def linear_solver_2d(
        self,
        constraints: List[LineConstraint],
        velocity_opt: torch.Tensor,
        speed_max: float,
        optimize_direction: bool = False,
    ) -> Tuple[int, torch.Tensor]:
        # Optimize closest point and outside circle if preferred velocity too fast (so normalize it to max speed).
        # Otherwise initialise the new velocity with preferred velocity.
        velocity_new = velocity_opt
        if torch.norm(velocity_opt) > speed_max:
            velocity_new = velocity_opt / torch.norm(velocity_opt) * speed_max

        for i in range(len(constraints)):
            # Result does not satisfy constraint i. Compute new optimal result.
            if torch.det(torch.stack((constraints[i].direction, constraints[i].point - velocity_new))) > 0:
                velocity_temp = velocity_new
                velocity_new = self.linear_program(
                    constraints=constraints,
                    i=i,
                    velocity_opt=velocity_opt,
                    speed_max=speed_max,
                    optimize_direction=optimize_direction
                )
                if velocity_new is None:
                    return i, velocity_temp
        return len(constraints), velocity_new

    def linear_solver_3d(
        self,
        constraints: List[LineConstraint],
        i_start: int,
        velocity_new: torch.Tensor,
        speed_max: float
    ) -> torch.Tensor:
        distance = 0

        for i in range(i_start, len(constraints)):

            # Result does not satisfy constraint of line i.
            if torch.det(torch.stack((constraints[i].direction, constraints[i].point - velocity_new))) > distance:
                constraints_p = []  # projected constraints
                for j in range(0, i):
                    determinant = torch.det(torch.stack((constraints[i].direction, constraints[j].direction)))

                    if torch.abs(determinant) < ORCA_EPS_NUMERIC:
                        # Line i and line j point in the same direction.
                        if constraints[i].direction.dot(constraints[j].direction) > 0:
                            continue
                        # Line i and line j point in opposite direction.
                        point = constraints[i].point + 0.5 * constraints[j].point
                    else:
                        in_dir = torch.stack(
                            (constraints[j].direction, constraints[i].point - constraints[j].point))
                        point = constraints[i].point + constraints[i].direction * torch.det(in_dir) / determinant

                    direction = constraints[j].direction - constraints[i].direction
                    direction = direction / torch.norm(direction)
                    constraints_p.append(ORCAEnvironment.LineConstraint(point=point, direction=direction))

                velocity_temp = velocity_new
                velocity_opt = torch.tensor([-constraints[i].direction[1], constraints[i].direction[0]])
                i_fail, velocity_new = self.linear_solver_2d(constraints_p, velocity_opt, speed_max,
                                                             optimize_direction=True)
                if i_fail < len(constraints_p):
                    velocity_new = velocity_temp
                distance = torch.det(torch.stack((constraints[i].direction, constraints[i].point - velocity_new)))

        return velocity_new

    @staticmethod
    def linear_program(
        constraints: List[LineConstraint],
        i: int,
        velocity_opt: torch.Tensor,
        speed_max: float,
        optimize_direction: bool = False,
    ) -> Union[torch.Tensor, None]:
        dot = constraints[i].point.dot(constraints[i].direction)
        discriminant = dot ** 2 + speed_max ** 2 - torch.norm(constraints[i].point) ** 2
        # Max speed circle fully invalidates line i.
        if discriminant < 0:
            return None

        t_left = -torch.sqrt(discriminant) - dot
        t_right = torch.sqrt(discriminant) - dot

        for j in range(len(constraints)):
            denominator = torch.det(torch.stack((constraints[i].direction, constraints[j].direction)))
            numerator = torch.det(
                torch.stack((constraints[j].direction, constraints[i].point - constraints[j].point)))

            # Lines (constraint lines) i and j are (almost) parallel.
            if torch.abs(denominator) < ORCA_EPS_NUMERIC:
                if numerator < 0.0:
                    return None
                continue

            # Line j bounds line i on the right or left.
            t = numerator / denominator
            t_right = min(t_right, t) if denominator >= 0.0 else t_right
            t_left = max(t_left, t) if denominator < 0.0 else t_left
            if t_left > t_right:
                return None

        if optimize_direction:
            if velocity_opt.dot(constraints[i].direction) > 0:
                velocity_new = constraints[i].point + t_right * constraints[i].direction
            else:
                velocity_new = constraints[i].point + t_left * constraints[i].direction

        else:
            t = constraints[i].direction.dot(velocity_opt - constraints[i].point)
            if t < t_left:
                velocity_new = constraints[i].point + t_left * constraints[i].direction
            elif t > t_right:
                velocity_new = constraints[i].point + t_right * constraints[i].direction
            else:
                velocity_new = constraints[i].point + t * constraints[i].direction
        return velocity_new

    @staticmethod
    def line_constraints(
        graph: Dict[str, torch.Tensor],
        k: int,
        self_id: str,
        other_ids: List[str],
        dt: float,
        agent_radius: float,
        agent_safe_dt: float
    ) -> List[LineConstraint]:
        # We only want to find a new velocity for the self agent, therefore we merely determine constraints
        # between the others agents in the scene and the self, not inter-ado constraints (as we cannot control
        # them anyway).
        constraints = []

        for other_id in other_ids:
            rel_pos = graph[f"{other_id}_{k}_{GK_POSITION}"] - graph[f"{self_id}_{k}_{GK_POSITION}"]
            rel_vel = graph[f"{self_id}_{k}_{GK_VELOCITY}"] - graph[f"{other_id}_{k}_{GK_VELOCITY}"]
            distance_sq = torch.norm(rel_pos) ** 2
            combined_radius = agent_radius + agent_radius
            combined_radius_sq = combined_radius ** 2

            # No collision.
            if distance_sq > combined_radius_sq:
                w = rel_vel - rel_pos / agent_safe_dt

                # Vector from cutoff center to relative velocity.
                w_length_sq = torch.norm(w) ** 2
                dot_1 = w.dot(rel_pos)

                if dot_1 < 0.0 and dot_1 ** 2 > combined_radius_sq * w_length_sq:
                    # Project on cut-off circle (normal direction).
                    direction = torch.tensor([w[1], -w[0]]) / torch.norm(w)
                    u = w * (combined_radius / agent_safe_dt / torch.norm(w) - 1)

                else:
                    # Project on legs.
                    leg = torch.sqrt(distance_sq - combined_radius_sq)

                    # Project on left or right leg, respectively, dependent on whether w points in the same or
                    # opposite directions as the relative position vector between both agents.
                    direction = torch.zeros(2)
                    if torch.det(torch.stack((rel_pos, w))) > 0:
                        direction[0] = rel_pos[0] * leg - rel_pos[1] * combined_radius
                        direction[1] = rel_pos[0] * combined_radius + rel_pos[1] * leg
                    else:
                        direction[0] = rel_pos[0] * leg + rel_pos[1] * combined_radius
                        direction[1] = -rel_pos[0] * combined_radius + rel_pos[1] * leg
                    direction = direction / distance_sq

                    u = direction * rel_vel.dot(direction) - rel_vel

            # Collision. Project on cut-off circle of time timeStep.
            else:
                # Vector from cutoff center to relative velocity.
                w = rel_vel - rel_pos / dt
                direction = torch.tensor([w[1], -w[0]]) / torch.norm(w)
                u = w * (combined_radius / dt / torch.norm(w) - 1)

            # usually 0.5 but we want to avoid interaction therefore 1.0
            point = graph[f"{self_id}_{k}_{GK_VELOCITY}"] + 0.5 * u
            constraints.append(ORCAEnvironment.LineConstraint(point=point, direction=direction))
        return constraints

    ###########################################################################
    # Operators ###############################################################
    ###########################################################################
    def _copy_ados(self, env_copy: 'GraphBasedEnvironment') -> 'GraphBasedEnvironment':
        for i in range(self.num_ados):
            ghosts_ado = self.ghosts_by_ado_index(ado_index=i)
            ado_id, _ = self.split_ghost_id(ghost_id=ghosts_ado[0].id)
            env_copy.add_ado(
                position=ghosts_ado[0].agent.position,  # uni-modal !
                velocity=ghosts_ado[0].agent.velocity,  # uni-modal !
                history=ghosts_ado[0].agent.history,  # uni-modal !
                time=self.time,
                weights=[ghost.weight for ghost in ghosts_ado],
                num_modes=self.num_modes,
                identifier=self.split_ghost_id(ghost_id=ghosts_ado[0].id)[0],
                goal=ghosts_ado[0].params[PK_GOAL],
            )
        return env_copy

    ###########################################################################
    # Simulation parameters ###################################################
    ###########################################################################
    @property
    def environment_name(self) -> str:
        return "orca"

    @property
    def is_multi_modal(self) -> bool:
        return False

    @property
    def is_deterministic(self) -> bool:
        return True

    @property
    def is_differentiable_wrt_ego(self) -> bool:
        return False
