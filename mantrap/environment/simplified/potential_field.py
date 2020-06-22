import logging
import math
import typing

import torch

import mantrap.agents
import mantrap.constants

from ..base.particle import ParticleEnvironment


class PotentialFieldEnvironment(ParticleEnvironment):
    """Simplified version of social forces environment class.

    The simplified model assumes static agents (ados) in the scene, having zero velocity (if not stated otherwise)
    and staying in this (non)-movement since no forces are applied. Hereby, the graph model is cut to the pure
    interaction between ego and ado, no inter-ado interaction and goal pulling force. Since the ados would not move
    at all without an ego agent in the scene, the interaction loss simply comes down the the distance of every position
    of the ados in time to their initial (static) position.

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
                         **particle_kwargs
                         ) -> typing.Tuple[typing.List[typing.List[mantrap.agents.IntegratorDTAgent]], torch.Tensor]:
        """Create particles from internal parameter distribution.

        In order to create parameters sample from the underlying parameter distributions, which are modelled
        as independent, uni-modal Gaussian distributions, individual for each ado. So build the distribution,
        sample N = num_particles values for each ado and parameter and create N copies of each ado (particle)
        storing the sampled parameter.

        :param num_particles: number of particles per ado.
        :param v0_dict: parameter v0 gaussian distribution (mean, variance) by ado_id, if None then gaussian
                        with mean, variance = `mantrap.constants.POTENTIAL_FIELD_V0_DEFAULT`
                        similarly for each ado.
        :return: list of N = num_particles for every ado in the scene.
        :return: probability (pdf) of each particle (num_ados, num_particles).
        """
        if v0_dict is None:
            v0_default, v0_variance = mantrap.constants.POTENTIAL_FIELD_V0_DEFAULT
            v0_dict = {ado_id: (v0_default, v0_variance) for ado_id in self.ado_ids}

        return super(PotentialFieldEnvironment, self).create_particles(num_particles, param_dicts={"v0": v0_dict})

    def simulate_particle(self,
                          particle: mantrap.agents.IntegratorDTAgent,
                          means_t: torch.Tensor,
                          ego_state_t: torch.Tensor = None
                          ) -> mantrap.agents.IntegratorDTAgent:
        """Forward simulate particle for one time-step (t -> t + 1).

        As described in the class description the potential field environment merely takes into account the
        repulsive force with respect to the ego, not to other ados, and introduces it as direct control input
        for the particle.

        :param particle: particle agent to simulate.
        :param means_t: means of positional and velocity distribution at time t (num_ados, 4).
        :param ego_state_t: ego/robot state at time t.
        """
        ego_impact = torch.zeros(2)
        v0 = max(particle.params["v0"], 1e-3)
        theta_attention = mantrap.constants.POTENTIAL_FIELD_MAX_THETA / 180.0 * math.pi

        if ego_state_t is not None:
            velocity = particle.velocity
            delta = ego_state_t[0:2] - particle.position

            # Only consider the effects of the robot, if inside attention angle.
            theta_self = torch.atan2(velocity[1], velocity[0])  # particle orientation
            theta_robot = torch.atan2(delta[1], delta[0])  # angle to robot
            theta_delta = theta_self - theta_robot
            if torch.abs(theta_delta) < theta_attention:
                ego_impact = - v0 * torch.sign(delta) * torch.exp(- torch.abs(delta))

        controls = particle.velocity + ego_impact
        particle.update(action=controls, dt=self.dt)
        logging.debug(f"particle {particle.id} impact = {ego_impact}")
        return particle

    ###########################################################################
    # Simulation parameters ###################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "potential_field"

    @property
    def is_differentiable_wrt_ego(self) -> bool:
        return True
