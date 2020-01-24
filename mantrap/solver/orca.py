from collections import namedtuple
import logging
from typing import List, Tuple, Union

import numpy as np

from mantrap.constants import eps_numeric, agent_speed_max, orca_agent_radius, orca_agent_safe_dt
from mantrap.agents import IntegratorDTAgent
from mantrap.simulation.simulation import Simulation
from mantrap.solver.solver import Solver


LineConstraint = namedtuple("LineConstraint", ["point", "direction"])


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

    def determine_ego_action(
        self,
        env: Simulation,
        agent_radius: float = orca_agent_radius,
        safe_dt: float = orca_agent_safe_dt,
        speed_max: float = agent_speed_max,
    ) -> Union[np.ndarray, None]:
        assert env.ego.__class__ == IntegratorDTAgent, "orca assumes robot to be holonomic and to have velocity input"

        # Find line constraints.
        constraints = self._line_constraints(env, agent_radius=agent_radius, agent_safe_dt=safe_dt)
        logging.info(f"ORCA constraints for ego [{env.ego.id}]: {constraints}")

        # Find new velocity based on line constraints. Preferred velocity would be the going from the current position
        # to the goal with maximal speed.
        velocity_preferred = self.goal - env.ego.position
        velocity_preferred = velocity_preferred / np.linalg.norm(velocity_preferred) * speed_max

        i_fail, velocity_new = self._linear_solver_2d(constraints, speed_max=speed_max, velocity_opt=velocity_preferred)
        if i_fail < len(constraints):
            velocity_new = self._linear_solver_3d(constraints, i_fail, speed_max=speed_max, velocity_new=velocity_new)

        return velocity_new

    def _linear_solver_2d(
        self, constraints: List[LineConstraint], velocity_opt: np.ndarray, speed_max: float, optimize_dir: bool = False
    ) -> Tuple[int, np.ndarray]:
        # Optimize closest point and outside circle if preferred velocity too fast (so normalize it to max speed).
        # Otherwise initialise the new velocity with preferred velocity.
        velocity_new = velocity_opt
        if np.linalg.norm(velocity_opt) > speed_max:
            velocity_new = velocity_opt / np.linalg.norm(velocity_opt) * speed_max

        for i in range(len(constraints)):
            # Result does not satisfy constraint i. Compute new optimal result.
            if np.linalg.det(np.vstack((constraints[i].direction, constraints[i].point - velocity_new))) > 0:
                velocity_temp = velocity_new
                velocity_new = self._linear_program(constraints, i, velocity_opt, speed_max, optimize_dir=optimize_dir)
                if velocity_new is None:
                    return i, velocity_temp
        return len(constraints), velocity_new

    def _linear_solver_3d(
        self, constraints: List[LineConstraint], i_start: int, velocity_new: np.ndarray, speed_max: float
    ) -> np.ndarray:
        distance = 0

        for i in range(i_start, len(constraints)):

            # Result does not satisfy constraint of line i.
            if np.linalg.det(np.vstack((constraints[i].direction, constraints[i].point - velocity_new))) > distance:
                constraints_p = []  # projected constraints
                for j in range(0, i):
                    determinant = np.linalg.det(np.vstack((constraints[i].direction, constraints[j].direction)))

                    if np.abs(determinant) < eps_numeric:
                        # Line i and line j point in the same direction.
                        if constraints[i].direction.dot(constraints[j].direction) > 0:
                            continue
                        # Line i and line j point in opposite direction.
                        point = constraints[i].point + 0.5 * constraints[j].point
                    else:
                        in_dir = np.vstack((constraints[j].direction, constraints[i].point - constraints[j].point))
                        point = constraints[i].point + constraints[i].direction * np.linalg.det(in_dir) / determinant

                    direction = constraints[j].direction - constraints[i].direction
                    direction = direction / np.linalg.norm(direction)
                    constraints_p.append(LineConstraint(point=point, direction=direction))

                velocity_temp = velocity_new
                velocity_opt = np.array([-constraints[i].direction[1], constraints[i].direction[0]])
                i_fail, velocity_new = self._linear_solver_2d(constraints_p, velocity_opt, speed_max, optimize_dir=True)
                if i_fail < len(constraints_p):
                    velocity_new = velocity_temp
                distance = np.linalg.det(np.vstack((constraints[i].direction, constraints[i].point - velocity_new)))

        return velocity_new

    @staticmethod
    def _linear_program(
        constraints: List[LineConstraint],
        i: int,
        velocity_opt: np.ndarray,
        speed_max: float,
        optimize_dir: bool = False,
    ) -> Union[np.ndarray, None]:
        dot = constraints[i].point.dot(constraints[i].direction)
        discriminant = dot ** 2 + speed_max ** 2 - np.linalg.norm(constraints[i].point) ** 2
        # Max speed circle fully invalidates line i.
        if discriminant < 0:
            return None

        t_left = -np.sqrt(discriminant) - dot
        t_right = np.sqrt(discriminant) - dot

        for j in range(len(constraints)):
            denominator = np.linalg.det(np.vstack((constraints[i].direction, constraints[j].direction)))
            numerator = np.linalg.det(
                np.vstack((constraints[j].direction, constraints[i].point - constraints[j].point))
            )

            # Lines (constraint lines) i and j are (almost) parallel.
            if np.abs(denominator) < eps_numeric:
                if numerator < 0.0:
                    return None
                continue

            # Line j bounds line i on the right or left.
            t = numerator / denominator
            t_right = min(t_right, t) if denominator >= 0.0 else t_right
            t_left = max(t_left, t) if denominator < 0.0 else t_left
            if t_left > t_right:
                return None

        if optimize_dir:
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
    def _line_constraints(env: Simulation, agent_radius: float, agent_safe_dt: float) -> List[LineConstraint]:
        constraints = []

        # We only want to find a new velocity for the ego robot, therefore we merely determine constraints between
        # the ados in the scene and the ego, not inter-ado constraints (as we cannot control them anyway).
        for ado in env.ados:
            rel_pos = ado.position - env.ego.position
            rel_vel = env.ego.velocity - ado.velocity
            distance_sq = np.linalg.norm(rel_pos) ** 2
            combined_radius = agent_radius + agent_radius
            combined_radius_sq = combined_radius ** 2

            # No collision.
            if distance_sq > combined_radius_sq:
                w = rel_vel - rel_pos / agent_safe_dt

                # Vector from cutoff center to relative velocity.
                w_length_sq = np.linalg.norm(w) ** 2
                dot_1 = w.dot(rel_pos)

                if dot_1 < 0.0 and dot_1 ** 2 > combined_radius_sq * w_length_sq:
                    # Project on cut-off circle (normal direction).
                    direction = np.array([w[1], -w[0]]) / np.linalg.norm(w)
                    u = w * (combined_radius / agent_safe_dt / np.linalg.norm(w) - 1)

                else:
                    # Project on legs.
                    leg = np.sqrt(distance_sq - combined_radius_sq)

                    # Project on left or right leg, respectively, dependent on whether w points in the same or
                    # opposite directions as the relative position vector between both agents.
                    direction = np.zeros(2)
                    if np.linalg.det(np.vstack((rel_pos, w))) > 0:
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
                w = rel_vel - rel_pos / env.dt
                direction = np.array([w[1], -w[0]]) / np.linalg.norm(w)
                u = w * (combined_radius / env.dt / np.linalg.norm(w) - 1)

            point = env.ego.velocity + 0.5 * u  # usually 0.5 but we want to avoid interaction there 1.0
            constraints.append(LineConstraint(point=point, direction=direction))
        return constraints