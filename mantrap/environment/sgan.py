import os
import sys
import typing

import attrdict
import torch
import torch.distributions

import mantrap.utility.io
import mantrap.utility.shaping

from .base import GraphBasedEnvironment


class SGAN(GraphBasedEnvironment):
    """SGAN from "Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks" (Gupta,  2018).

    The SGAN model is a GAN-based model for pedestrian prediction. As it is not conditioned on the robot, it is
    not of use for robot trajectory optimization here. However, it still can be utilized as an independent, but
    accurate pedestrian prediction model for evaluation purposes.

    As a consequence, merely the trajectory sampling functions are implemented here. Since SGAN is based
    on a GAN, it merely outputs an empirical distribution, therefore mean values (for prediction-function)
    cannot be evaluated as well, without averaging over many samples.
    """
    def __init__(
        self,
        ego_position: torch.Tensor = None,
        ego_velocity: torch.Tensor = torch.zeros(2),
        ego_history: torch.Tensor = None,
        ego_type: mantrap.agents.base.DTAgent.__class__ = mantrap.agents.DoubleIntegratorDTAgent,
        dt: float = mantrap.constants.ENV_DT_DEFAULT,
        **env_kwargs
    ):
        # Initialize environment mother class.
        super(SGAN, self).__init__(ego_position, ego_velocity, ego_history, ego_type, dt=dt, **env_kwargs)

        # Load SGAN generator model.
        model_path = os.path.join(self.module_os_path(), mantrap.constants.SGAN_MODEL)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self._sgan = self.get_generator(checkpoint)

    ###########################################################################
    # Scene ###################################################################
    ###########################################################################
    def add_ado(
            self,
            position: torch.Tensor,
            velocity: torch.Tensor = torch.zeros(2),
            history: torch.Tensor = None,
            **ado_kwargs
    ) -> mantrap.agents.IntegratorDTAgent:
        """Add ado (i.e. non-robot) agent to environment as single integrator.

        While the ego is added to the environment during initialization, the ado agents have to be added afterwards,
        individually. To do so initialize single integrator agent using its state vectors, namely position, velocity
        and its state history. The ado id, color and other parameters can either be passed using the ado_kwargs
        option or are created automatically during the agent's initialization.

        After initialization check whether the given states are valid, i.e. do  not pass the internal environment
        bounds, e.g. that they are in the given 2D space the environment is defined in.

        The SGAN model is trained to predict accurately, iff the agent has some history > 1, therefore if no
        history is given build a custom history by stacking the given (position, velocity) state over multiple
        time-steps. If zero history should be enforced, pass a non None history argument.

        :param position: ado initial position (2D).
        :param velocity: ado initial velocity (2D).
        :param history: ado state history (if None then just stacked current state).
        :param ado_kwargs: addition kwargs for ado initialization.
        """
        if history is None or history.shape[0] == 1:
            position, velocity = position.float(), velocity.float()
            history = torch.stack([torch.cat(
                (position + velocity * self.dt * t, velocity, torch.ones(1) * self.time + self.dt * t))
                for t in range(-mantrap.constants.TRAJECTRON_DEFAULT_HISTORY_LENGTH, 1)
            ])
        return super(SGAN, self).add_ado(position, velocity=velocity, history=history, **ado_kwargs)

    ###########################################################################
    # Prediction - Samples ####################################################
    ###########################################################################
    def sample_w_trajectory(self, ego_trajectory: torch.Tensor, num_samples: int = 1
                            ) -> typing.Union[torch.Tensor, None]:
        """Predict the ado path samples based conditioned on robot trajectory.

        As described above the SGAN predictions are not conditioned on the ego trajectory. Consequently,
        this method is equal to the `sample_wo_ego()` method (including the ego agent).

        :param ego_trajectory: ego trajectory (prediction_horizon + 1, 5).
        :param num_samples: number of samples to return.
        :return: predicted ado paths (num_ados, num_samples, prediction_horizon+1, num_modes=1, 2).
                 if no ado in scene, return None instead.
        """
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, pos_and_vel_only=True)
        assert self.sanity_check(check_ego=True)
        t_horizon = ego_trajectory.shape[0] - 1

        self.add_ado(self.ego.position, velocity=self.ego.velocity, history=self.ego.history)
        samples = self.sample_wo_ego(t_horizon=t_horizon, num_samples=num_samples)
        self._ados = self.ados[:-1]
        return samples[:-1]

    def sample_wo_ego(self, t_horizon: int, num_samples: int = 1) -> typing.Union[torch.Tensor, None]:
        """Predict the unconditioned ado path samples (i.e. if no robot would be in the scene).

        For prediction simply call the SGAN generator model and sample trajectories from it.

        :param t_horizon: prediction horizon, number of discrete time-steps.
        :param num_samples: number of samples to return.
        :return: predicted ado paths (num_ados, num_samples, prediction_horizon+1, num_modes=1, 2).
                 if no ado in scene, return None instead.
        """
        assert 0 < t_horizon < 8  # 8 = sample length sgan was trained on

        # Get ado histories and relative histories (aka velocities for dt=1.0)
        # SGAN shape: (time_steps, num_ados, 2 = x, y)
        ado_histories = torch.stack([ado.history[:, 0:2] for ado in self.ados], dim=1)
        ado_histories_rel = self.absolute_to_relative(ado_histories)
        ado_start_pos = ado_histories[-1, :, 0:2]
        seq_start_end = torch.tensor([0, self.num_ados]).view(1, 2)

        # Predict trajectory samples from SGAN generator.
        samples = torch.zeros((self.num_ados, num_samples, t_horizon + 1, 1, 2))
        for n in range(num_samples):
            ado_samples_rel = self._sgan(ado_histories, ado_histories_rel, seq_start_end)
            ado_samples = self.relative_to_abs(ado_samples_rel, start_pos=ado_start_pos)

            # (time-steps, num_ados, 2) -> (num_ados, num_samples, time_steps, 2).
            samples[:, n, :, 0, :] = ado_samples[:t_horizon + 1, :, :].permute(1, 0, 2)

        assert mantrap.utility.shaping.check_ado_samples(samples, t_horizon=t_horizon + 1, num_samples=num_samples)
        return samples

    ###########################################################################
    # Simulation graph ########################################################
    ###########################################################################
    def _compute_distributions(self, ego_trajectory: torch.Tensor, vel_dist: bool = True, **kwargs
                               ) -> typing.Dict[str, torch.distributions.Distribution]:
        raise NotImplementedError

    def _compute_distributions_wo_ego(self, t_horizon: int, vel_dist: bool = True, **kwargs
                                      ) -> typing.Dict[str, torch.distributions.Distribution]:
        raise NotImplementedError

    ###########################################################################
    # SGAN-specific methods ###################################################
    ###########################################################################
    @staticmethod
    def absolute_to_relative(trajectory: torch.Tensor) -> torch.Tensor:
        t_horizon, num_agents, dim = trajectory.shape
        trajectory_rel = trajectory - trajectory[0, :, :].unsqueeze(dim=0)
        trajectory_rel = trajectory_rel[1:, :, :] - trajectory_rel[:-1, :, :]
        return torch.cat((torch.zeros(1, num_agents, dim), trajectory_rel), dim=0)

    @staticmethod
    def relative_to_abs(rel_trajectory: torch.Tensor, start_pos: torch.Tensor) -> torch.Tensor:
        rel_trajectory = rel_trajectory.permute(1, 0, 2)
        displacement = torch.cumsum(rel_trajectory, dim=1)
        start_pos = torch.unsqueeze(start_pos, dim=1)
        abs_trajectory = displacement + start_pos
        return abs_trajectory.permute(1, 0, 2)

    def get_generator(self, checkpoint):
        self.import_modules()
        from sgan.models import TrajectoryGenerator
        args = attrdict.AttrDict(checkpoint['args'])
        generator = TrajectoryGenerator(
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            embedding_dim=args.embedding_dim,
            encoder_h_dim=args.encoder_h_dim_g,
            decoder_h_dim=args.decoder_h_dim_g,
            mlp_dim=args.mlp_dim,
            num_layers=args.num_layers,
            noise_dim=args.noise_dim,
            noise_type=args.noise_type,
            noise_mix_type=args.noise_mix_type,
            pooling_type=args.pooling_type,
            pool_every_timestep=args.pool_every_timestep,
            dropout=args.dropout,
            bottleneck_dim=args.bottleneck_dim,
            neighborhood_size=args.neighborhood_size,
            grid_size=args.grid_size,
            batch_norm=args.batch_norm)
        generator.load_state_dict(checkpoint['g_state'])
        generator.cpu()
        generator.train()
        return generator

    @staticmethod
    def import_modules():
        if SGAN.module_os_path() not in sys.path:
            sys.path.insert(0, SGAN.module_os_path())

    @staticmethod
    def module_os_path() -> str:
        module_path = mantrap.utility.io.build_os_path("third_party/sgan", make_dir=False, free=False)
        assert os.path.isdir(module_path)
        return module_path

    ###########################################################################
    # Simulation parameters ###################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "sgan"

    @property
    def num_modes(self) -> int:
        return 1

    @property
    def is_differentiable_wrt_ego(self) -> bool:
        return False
