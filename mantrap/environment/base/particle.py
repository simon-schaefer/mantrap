import abc
import typing

import torch
import torch.distributions

import mantrap.agents
import mantrap.constants
import mantrap.utility.shaping

from .graph_based import GraphBasedEnvironment


class ParticleEnvironment(GraphBasedEnvironment, abc.ABC):

    @abc.abstractmethod
    def create_particles(self,
                         num_particles: int,
                         param_dicts: typing.Dict[str, typing.Dict[str, typing.Dict]] = None,
                         const_dicts: typing.Dict[str, typing.Dict[str, typing.Any]] = None,
                         **particle_kwargs
                         ) -> typing.Tuple[typing.List[typing.List[mantrap.agents.IntegratorDTAgent]], torch.Tensor]:
        """Create particles from internal parameter distribution.

        In order to create parameters sample from the underlying parameter distributions, which are modelled
        as independent, uni-modal Gaussian distributions, individual for each ado. So build the distribution,
        sample N = num_particles values for each ado and parameter and create N copies of each ado (particle)
        storing the sampled parameter.

        Attention: Not the same convention over all input parameters !

        :param num_particles: number of particles per ado.
        :param param_dicts: parameter dictionaries for varying parameters between particles.
                            {param_name: {ado_id: (mean, variance)}, ....}
        :param const_dicts: dictionary mapping ado-wise constant parameters to parameter name.
                            {ado_id: {param_name: values}, ....}
        :return: list of N = num_particles for every ado in the scene.
        :return: probability (pdf) of each particle (num_ados, num_particles).
        """
        particles = []
        num_params = len(param_dicts.keys())
        samples = torch.zeros((self.num_ados, num_params, num_particles))
        pdfs = torch.zeros((self.num_ados, num_params, num_particles))
        for m_ado, ado in enumerate(self.ados):
            ado_id = ado.id

            # Check (or replace) constant parameters dict object.
            const_params = {}
            if const_dicts is not None:
                assert ado_id in const_dicts.keys()
                const_params = const_dicts[ado_id]

            # Build and sample from parameter distribution for each parameter assigned to the current ado.
            for ip, (p_key, p_dict) in enumerate(param_dicts.items()):
                assert all([ado_id in p_dict.keys() for ado_id in self.ado_ids])
                p_values = p_dict[ado_id]
                assert len(p_values) == 2  # (mean, variance) of distribution
                loc, scale = p_values
                distribution = torch.distributions.Normal(loc=loc, scale=scale)
                sample_n = distribution.sample((num_particles, ))
                samples[m_ado, ip, :] = sample_n
                pdfs[m_ado, ip, :] = distribution.cdf(sample_n)

            # Initialize ado particles. Unfortunately, this operation cannot be further batched  since the
            # particle initialization __init__ call does only allow to create one class object.
            ado_particles = []
            for n in range(num_particles):
                particle_params = {p_key: samples[m_ado, ip, n] for ip, p_key in enumerate(param_dicts.keys())}
                particle = mantrap.agents.IntegratorDTAgent(
                    position=ado.position, velocity=ado.velocity, history=ado.history, time=self.time,
                    identifier=ado_id, color=ado.color, **particle_params, **const_params, **particle_kwargs
                )
                ado_particles.append(particle)
            particles.append(ado_particles)

        # Under the assumption of independence of parameters (which is given by independent sampling here, just
        # assuming the parameters itself are independent), we calculate each particles pdf by multiplying their
        # parameters probability densities.
        return particles, torch.prod(pdfs, dim=1)

    @abc.abstractmethod
    def simulate_particle(self,
                          particle: mantrap.agents.IntegratorDTAgent,
                          means_t: torch.Tensor,
                          ego_state_t: torch.Tensor = None
                          ) -> mantrap.agents.IntegratorDTAgent:
        """Forward simulate particle for one time-step (t -> t + 1).

        :param particle: particle agent to simulate.
        :param means_t: means of positional and velocity distribution at time t (num_ados, 4).
        :param ego_state_t: ego/robot state at time t.
        """
        raise NotImplementedError

    ###########################################################################
    # Simulation Graph over time-horizon ######################################
    ###########################################################################
    def _compute_distributions(self, ego_trajectory: typing.Union[typing.List, torch.Tensor],
                               num_particles: int = mantrap.constants.ENV_NUM_PARTICLES,
                               **kwargs
                               ) -> typing.Dict[str, torch.distributions.Distribution]:
        """Build a connected graph based on the ego's trajectory.

        The graph should span over the time-horizon of the length of the ego's trajectory and contain the
        velocity distribution of every ado in the scene as well as the ego's states itself. When
        possible the graph should be differentiable, such that finding some gradient between the outputted ado
        states and the inputted ego trajectory is determinable.

        Iterative environment build up a prediction over a time horizon > 1 stepwise, i.e. predicting the first
        step t0 -> t1, then plugging in the results to the next prediction t1 -> t2, etc, until tN. Also they are
        usually (at least not the ones regarded within the scope of this project) not conditioned on the presence
        of some ego agent, so that instead of a trajectory simply a list of None can be passed, in order to build
        a graph without an ego in the scene.

        The velocity distribution thereby is computed by averaging over a bunch of simulated particles, merged
        to a single uni-modal gaussian for simplicity. Since the models only predict the velocity distribution in
        order to form the full state the velocity has to be computed from that. Since the ados underlie single
        integrator dynamics the velocity just is the difference of position times 1/dt. For two uni-modal
        Gaussian distributions (current, next) this results to a non-central chi-squared distribution in general,
        however assuming independence we get a normal distribution with

        .. math:: \\mu_{X-Y} = \\mu_X - \\mu_Y

        .. math:: \\sigma_{X-Y}^2 = \\sigma_X^2 + \\sigma_Y^2

        .. math:: c \\cdot N(\\mu, \\sigma^2) \\sim N(c \\cdot \\mu, c^2 \\cdot \\sigma^2)

        note: however the mean values are the same anyway, which are used for computing the subsequent distributions
        (https://stats.stackexchange.com/questions/186463/distribution-of-difference-between-two-normal-distributions).

        :param ego_trajectory: ego's trajectory (t_horizon, 5).
        :return: ado_id-keyed velocity distribution dictionary for times [0, t_horizon].
        """
        if not all([x is None for x in ego_trajectory]):
            assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, pos_and_vel_only=True)
        t_horizon = len(ego_trajectory) - 1  # works for list and torch.Tensor (!)

        # Create particles using the environment-specific method.
        particles, particle_pdf = self.create_particles(num_particles=num_particles, **kwargs)
        particle_pdf = (particle_pdf / torch.norm(particle_pdf, dim=0)).unsqueeze(dim=2).detach()  # normalize

        # For each time-step predict the next distribution by simulating several particles and averaging
        # them to a uni-modal gaussian distribution.
        _, ado_states = self.states()

        mus = torch.zeros((self.num_ados, t_horizon, 1, 4))  # ados, t_horizon, modes, 4 (= position + velocity)
        mus[:, 0, 0, :] = ado_states[:, 0:4]
        sigmas = torch.zeros((self.num_ados, t_horizon, 1, 2))  # velocity only
        sigmas[:, 0, 0, :] = torch.ones((self.num_ados, 2)) * mantrap.constants.ENV_VAR_INITIAL
        for t in range(t_horizon - 1):
            ego_state_t = ego_trajectory[t]
            ado_states_t = mus[:, t, 0, :]
            velocities_t = torch.zeros((self.num_ados, num_particles, 2))

            # Simulate and update the particles for each ado in the scene and the current time-step.
            for m_ado in range(self.num_ados):
                particles_ado = particles[m_ado]
                for m_particle, particle in enumerate(particles_ado):
                    particles[m_ado][m_particle] = self.simulate_particle(particle, ado_states_t, ego_state_t)
                    velocities_t[m_ado, m_particle, :] = particles[m_ado][m_particle].velocity

            # By adding a tiny amount of white gaussian noise we avoid troubles with zero variance
            # (e.g. in Potential Field Environment with uni-directional interactions).
            velocities_t += torch.rand(velocities_t.shape) * mantrap.constants.ENV_PARTICLE_NOISE

            # Estimate the overall velocity distribution of every ado in the next time-step, by averaging over
            # all updated particles. Then compute the mean of the velocity distribution from that.
            # Weight the position estimate of each particle with their probability occurring in the initial
            # distribution they have been sampled from.
            velocities_t_pdf = velocities_t * particle_pdf

            mus[:, t + 1, 0, 2:4] = torch.mean(velocities_t_pdf, dim=1)
            mus[:, t + 1, 0, 0:2] = mus[:, t, 0, 0:2] + mus[:, t, 0, 2:4] * self.dt  # single integrator (!)
            sigmas[:, t + 1, 0, :] = torch.var(velocities_t_pdf, dim=1)

        # Transform mus and sigmas to velocity gaussian distribution objects dictionary
        # (hint: same order of ado_ids and ados() have been ensured in sanity_check() !).
        dist_dict = {ado_id: torch.distributions.Normal(loc=mus[m_ado, :, :, 2:4], scale=sigmas[m_ado, :, :, :])
                     for m_ado, ado_id in enumerate(self.ado_ids)}

        return dist_dict

    def _compute_distributions_wo_ego(self, t_horizon: int, **kwargs
                                      ) -> typing.Dict[str, torch.distributions.Distribution]:
        """Build a dictionary of velocity distributions for every ado as it would be without the presence
        of a robot in the scene.

        To simplify the implementation (make it more compact) to compute the distributions without the presence
        of a robot in the scene, compute the one conditioned on the robot with a `None` trajectory.

        :param t_horizon: number of prediction time-steps.
        :kwargs: additional graph building arguments.
        :return: ado_id-keyed velocity distribution dictionary for times [0, t_horizon].
        """
        return self._compute_distributions(ego_trajectory=[None] * (t_horizon + 1), **kwargs)

    ###########################################################################
    # Simulation parameters ###################################################
    ###########################################################################
    @property
    def num_modes(self) -> int:
        return 1
