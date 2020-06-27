import collections
import typing

import torch

import mantrap.constants

from ..base import TrajOptSolver


class ORCASolver(TrajOptSolver):
    """ORCA-based planning algorithm.

    Implementation of a planning algorithm according to 'Reciprocal n-body Collision Avoidance' by Jur van den Berg,
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

    LineConstraint = collections.namedtuple("LineConstraint", ["point", "direction"])

    def optimize_core(
        self,
        z0: torch.Tensor,
        ado_ids: typing.List[str],
        tag: str = mantrap.constants.TAG_OPTIMIZATION,
        agent_radius: float = mantrap.constants.ORCA_AGENT_RADIUS,
        safe_time: float = mantrap.constants.ORCA_SAFE_TIME,
        **solver_kwargs
    ) -> typing.Tuple[torch.Tensor, typing.Dict[str, torch.Tensor]]:
        """Optimization function for single core to find optimal z-vector.

        Given some initial value `z0` find the optimal allocation for z with respect to the internally defined
        objectives and constraints. This function is executed in every thread in parallel, for different initial
        values `z0`. To simplify optimization not all agents in the scene have to be taken into account during
        the optimization but only the ones with ids defined in `ado_ids`.

        Implementation of the ORCA algorithm for collision-free pedestrian avoidance. This solver only takes into
        account the current states, not any future state, which makes it efficient to solve, but also not very
        much anticipative.

        :param z0: initial value of optimization variables.
        :param tag: name of optimization call (name of the core).
        :param ado_ids: identifiers of ados that should be taken into account during optimization.
        :param agent_radius: ado collision-avoidance safety radius [m].
        :param safe_time: time interval of guaranteed no collisions [s].
        :returns: z_opt (optimal values of optimization variable vector)
                  optimization_log (logging dictionary for this optimization = self.log)
        """
        ego_state, ado_states = self.env.states()
        ego_state = ego_state.detach()
        ado_states = ado_states.detach()
        _, ego_v_max = self.env.ego.speed_limits()
        controls = torch.zeros((self.planning_horizon, 2))

        # ORCA environment loop. In order to function properly for ORCA the re-planning time-interval (= dt) has
        # to be quite low, < 0.1 s. Since the simulation runs quite fast, we can split the actual simulation
        # time-step in smaller sub-steps. This improves the result while being independent from the user-defined
        # simulation time-step, at a comparably low computational cost.
        assert self.env.dt <= 0.1

        for t in range(self.planning_horizon):
            # Find set of line constraint for the robot to be safe. Important here is that uni-modality is assumed.
            # Since only the robot's safety is to be determined, no inter-ado actions are taken into account
            # explicitly.
            constraints = self.line_constraints(ego_state=ego_state, ado_states=ado_states,
                                                dt=self.env.dt, agent_radius=agent_radius, agent_safe_dt=safe_time)

            # Find new velocity based on line constraints. Preferred velocity would be the going from the
            # current position to the goal directly with maximal speed. However when the goal has been reached
            # set the preferred velocity to zero (not original ORCA).
            # Assume the goal to be 5 meters in the direction of the initial orientation.
            goal_direction = self.goal - ego_state[0:2]
            vel_preferred = goal_direction / torch.norm(goal_direction) * ego_v_max

            # Solve the constrained optimization problem defined above.
            i_fail, vel_new = self.linear_solver_2d(constraints, speed_max=ego_v_max, velocity_opt=vel_preferred)
            if i_fail < len(constraints):
                vel_new = self.linear_solver_3d(constraints, i_fail, speed_max=ego_v_max, velocity_new=vel_new)

            # Update states for sub-time-step, assuming constant motion of ados.
            ego_state = self.env.ego.dynamics(ego_state, action=vel_new, dt=self.env.dt)
            for ado in self.env.ados:
                m_ado = self.env.index_ado_id(ado_id=ado.id)
                ado_action = ado_states[m_ado, 2:4]
                ado_states[m_ado, :] = ado.dynamics(ado_states[m_ado, :], action=ado_action, dt=self.env.dt)
            controls[t, :] = vel_new

        return controls, self.logger.log

    ###########################################################################
    # ORCA specific solver methods ############################################
    ###########################################################################
    def linear_solver_2d(
        self,
        constraints: typing.List[LineConstraint],
        velocity_opt: torch.Tensor,
        speed_max: float,
        optimize_direction: bool = False,
    ) -> typing.Tuple[int, torch.Tensor]:
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
        constraints: typing.List[LineConstraint],
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

                    if torch.abs(determinant) < 1e-3:
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
                    constraints_p.append(ORCASolver.LineConstraint(point=point, direction=direction))

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
        constraints: typing.List[LineConstraint],
        i: int,
        velocity_opt: torch.Tensor,
        speed_max: float,
        optimize_direction: bool = False,
    ) -> typing.Union[torch.Tensor, None]:
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
            if torch.abs(denominator) < 1e-3:
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
    def line_constraints(ego_state: torch.Tensor, ado_states: torch.Tensor,
                         dt: float, agent_radius: float, agent_safe_dt: float) -> typing.List[LineConstraint]:
        constraints = []

        for m_ado in range(ado_states.shape[0]):
            rel_pos = ado_states[m_ado, 0:2] - ego_state[0:2]
            rel_vel = ego_state[2:4] - ado_states[m_ado, 2:4]
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
            point = ego_state[2:4] + 0.5 * u
            constraints.append(ORCASolver.LineConstraint(point=point, direction=direction))
        return constraints

    ###########################################################################
    # Problem formulation - Reset #############################################
    ###########################################################################
    @staticmethod
    def module_defaults() -> typing.Union[typing.List[typing.Tuple], typing.List]:
        return []

    ###########################################################################
    # Solver properties #######################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "orca"
