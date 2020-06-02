import typing

import torch

import mantrap.agents
import mantrap.constants
import mantrap.utility.shaping

from .base.particle import ParticleEnvironment


class SocialForcesEnvironment(ParticleEnvironment):
    """Social Forces Simulation.

    Pedestrian Dynamics based on to "Social Force Model for Pedestrian Dynamics" (D. Helbling, P. Molnar). The idea
    of Social Forces is to determine interaction forces by taking into account the following entities:

    *Goal force*:
    Each ado has a specific goal state/position in mind, to which it moves to. The goal pulling force
    is modelled as correction term between the direction vector of the current velocity and the goal direction
    (vector between current position and goal).

    .. math:: F_{goal} = 1 / tau_{a} (v_a^0 e_a - v_a)

    *Interaction force*:
    For modelling interaction between multiple agents such as avoiding collisions each agent
    has an with increasing distance exponentially decaying repulsion field. Together with the scalar product of the
    velocity of each agent pair (in order to not alter close but non interfering agents, e.g. moving parallel to
    each other) the interaction term is constructed.

    .. math:: V_{aB} (x) = V0_a exp(−x / \\sigma_a)
    .. math:: F_{interaction} = - grad_{r_{ab}} V_{aB}(||r_{ab}||)

    Although theoretically  a force introduces an acceleration as the ado's control input (double not single
    integrator dynamics), accelerations are simplified to act as velocities, which is reasonable due to the fast
    reaction time of the pedestrian, which is way faster than the environment sampling rate.
    """

    ###########################################################################
    # Particle simulations ####################################################
    ###########################################################################
    def create_particles(self,
                         num_particles: int,
                         v0_dict: typing.Dict[str, typing.Tuple[float, float]] = None,
                         sigma_dict: typing.Dict[str, typing.Tuple[float, float]] = None,
                         tau: float = mantrap.constants.SOCIAL_FORCES_DEFAULT_TAU,
                         **unused
                         ) -> typing.Tuple[typing.List[typing.List[mantrap.agents.IntegratorDTAgent]], torch.Tensor]:
        """Create particles from internal parameter distribution.

        In order to create parameters sample from the underlying parameter distributions, which are modelled
        as independent, uni-modal Gaussian distributions, individual for each ado. So build the distribution,
        sample N = num_particles values for each ado and parameter and create N copies of each ado (particle)
        storing the sampled parameter.

        :param num_particles: number of particles per ado.
        :param v0_dict: parameter v0 gaussian distribution (mean, variance) by ado_id, if None then gaussian
                        with mean = `mantrap.constants.SOCIAL_FORCES_DEFAULT_V0` and variance = mean/4,
                        similarly for each ado.
        :param sigma_dict: parameter sigma gaussian distribution (similar to `v0_dict`).
        :param tau: tau parameter, by default `mantrap.constants.SOCIAL_FORCES_DEFAULT_TAU`,
                    which has to be shared over all agents.
        :return: list of N = num_particles for every ado in the scene.
        :return: probability (pdf) of each particle (num_ados, num_particles).
        """
        if v0_dict is None:
            v0_default = mantrap.constants.SOCIAL_FORCES_DEFAULT_V0
            v0_dict = {ado_id: (v0_default, v0_default / 4) for ado_id in self.ado_ids}
        if sigma_dict is None:
            sigma_default = mantrap.constants.SOCIAL_FORCES_DEFAULT_SIGMA
            sigma_dict = {ado_id: (sigma_default, sigma_default / 4) for ado_id in self.ado_ids}
        goal_dict = {ado.id: {"goal": ado.params["goal"]} for ado in self.ados}
        return super(SocialForcesEnvironment, self).create_particles(
            num_particles, param_dicts={"v0": v0_dict, "sigma": sigma_dict}, const_dicts=goal_dict, tau=tau,
        )

    def simulate_particle(self,
                          particle: mantrap.agents.IntegratorDTAgent,
                          means_t: torch.Tensor,
                          ego_state_t: torch.Tensor = None
                          ) -> mantrap.agents.IntegratorDTAgent:
        """Forward simulate particle for one time-step (t -> t + 1).

        Use the social forces equations writen in the class description for updating the particles. Thereby take
        into account repulsive forces from both the robot and other ados as well as a pulling force between
        the particle and its "goal" state.

        :param particle: particle agent to simulate.
        :param means_t: means of positional and velocity distribution at time t (num_ados, 4).
        :param ego_state_t: ego/robot state at time t.
        """

        # Repulsive force introduced by every other agent (depending on relative position and (!) velocity).
        def _repulsive_force(
            alpha_position: torch.Tensor,
            beta_position: torch.Tensor,
            alpha_velocity: torch.Tensor,
            beta_velocity: torch.Tensor,
        ) -> torch.Tensor:

            # Relative properties and their norms.
            relative_distance = torch.sub(alpha_position, beta_position)
            if not relative_distance.requires_grad:
                relative_distance.requires_grad = True
            else:
                relative_distance.retain_grad()  # get gradient without being leaf node
            relative_velocity = torch.sub(alpha_velocity, beta_velocity)

            norm_relative_distance = torch.norm(relative_distance)
            norm_relative_velocity = torch.norm(relative_velocity)
            norm_diff_position = torch.sub(relative_distance, relative_velocity * self.dt).norm()

            # Alpha-Beta potential field.
            b1 = torch.add(norm_relative_distance, norm_diff_position)
            b2 = self.dt * norm_relative_velocity
            b = 0.5 * torch.sqrt(torch.sub(torch.pow(b1, 2), torch.pow(b2, 2)))
            v = p_v0 * torch.exp(-b / p_sigma)

            # The repulsive force between agents is the negative gradient of the other (beta -> alpha)
            # potential field. Therefore subtract the gradient of V w.r.t. the relative distance.
            force = torch.autograd.grad(v, relative_distance, create_graph=True)[0]
            if torch.any(torch.isnan(force)):  # TODO: why is nan rarely occurring here and how to deal with that ?
                return torch.zeros(2)
            else:
                return force

        p_pos = particle.position
        p_vel = particle.velocity

        p_v0 = particle.params["v0"]
        p_sigma = particle.params["sigma"]
        p_tau = particle.params["tau"]
        p_goal = particle.params["goal"]

        # Destination force - Force pulling the ado to its assigned goal position.
        direction = torch.sub(p_goal, p_pos)
        goal_distance = torch.norm(direction)
        if goal_distance.item() < mantrap.constants.SOCIAL_FORCES_MAX_GOAL_DISTANCE:
            destination_force = torch.zeros(2)
        else:
            direction = torch.div(direction, goal_distance)
            speed = torch.norm(p_vel)
            destination_force = torch.sub(direction * speed, p_vel) * 1 / p_tau

        # Interactive force - Repulsive potential field by every other agent.
        repulsive_force = torch.zeros(2)
        for m_ado, ado_id in enumerate(self.ado_ids):
            if particle.id == ado_id:  # particles from the same parent agent dont repulse each other
                continue
            distance = torch.sub(p_pos, means_t[m_ado, 0:2])
            if torch.norm(distance) > mantrap.constants.SOCIAL_FORCES_MAX_INTERACTION_DISTANCE:
                continue
            else:
                v_grad = _repulsive_force(p_pos, means_t[m_ado, 0:2], p_vel, means_t[m_ado, 2:4])
            repulsive_force = torch.sub(repulsive_force, v_grad)

        # Interactive force w.r.t. ego - Repulsive potential field.
        ego_force = torch.zeros(2)
        if ego_state_t is not None:
            v_grad = _repulsive_force(p_pos, ego_state_t[0:2], p_vel, ego_state_t[2:4])
            ego_force = - v_grad

        # Update particle given the previously derived "forces".
        controls = destination_force + repulsive_force + ego_force
        particle.update(controls, dt=self.dt)
        return particle

    ###########################################################################
    # Scene ###################################################################
    ###########################################################################
    def add_ado(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor = torch.zeros(2),
        history: torch.Tensor = None,
        goal: torch.Tensor = torch.zeros(2),
        **ado_kwargs
    ) -> mantrap.agents.IntegratorDTAgent:
        goal = goal.float().detach()
        return super(SocialForcesEnvironment, self).add_ado(position, velocity, history, goal=goal, **ado_kwargs)

    ###########################################################################
    # Simulation properties ###################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "social_forces"

    @property
    def is_differentiable_wrt_ego(self) -> bool:
        return True
