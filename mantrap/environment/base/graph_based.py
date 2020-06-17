import abc
import logging
import typing

import numpy as np
import torch
import torch.distributions

import mantrap.agents
import mantrap.constants
import mantrap.utility.shaping


class GraphBasedEnvironment(abc.ABC):
    """General environment engine for obstacle-free, interaction-aware, probabilistic and multi-modal agent
    environments. As used in a robotics use-case the environment separates between the ego-agent (the robot) and
    ado-agents (other agents in the scene which are not the robot).

    The internal states basically are the states of the ego and ados and can only be changed by either using the
    `step()` or `step_reset()` function, which simulate how the environment reacts based on some action performed by
    the ego or resets it directly to some given states.

    The simulated world is two-dimensional and defined in the area limited by the passed `x_axis` and `y_axis`. It has
    a constant environment time-step `dt`.
    """

    ###########################################################################
    # Initialization ##########################################################
    ###########################################################################
    def __init__(
        self,
        ego_position: torch.Tensor = None,
        ego_velocity: torch.Tensor = torch.zeros(2),
        ego_history: torch.Tensor = None,
        ego_type: mantrap.agents.base.DTAgent.__class__ = mantrap.agents.DoubleIntegratorDTAgent,
        x_axis: typing.Tuple[float, float] = mantrap.constants.ENV_X_AXIS_DEFAULT,
        y_axis: typing.Tuple[float, float] = mantrap.constants.ENV_Y_AXIS_DEFAULT,
        dt: float = mantrap.constants.ENV_DT_DEFAULT,
        time: float = 0.0,
        config_name: str = mantrap.constants.CONFIG_UNKNOWN
    ):
        """Graph-Based environment initialization.

        :param ego_type: agent class of ego agent (should be agent child-class).
        :param ego_position: initial ego/robot position in 2D, must be defined if `ego_type` is defined.
        :param ego_velocity: initial ego/robot velocity in 2D, zero by default.
        :param ego_history: initial ego state history, None (only current state) by default.
        :param x_axis: environment environment limitation in x-direction.
        :param y_axis: environment environment limitation in y-direction.
        :param dt: environment time-step [s].
        :param config_name: configuration name of initialized environment (for logging purposes only).
        """
        assert x_axis[0] < x_axis[1]
        assert y_axis[0] < y_axis[1]
        assert dt > 0.0

        if ego_position is not None:
            assert ego_type is not None
            self._ego = ego_type(ego_position, velocity=ego_velocity, history=ego_history,
                                 time=time, is_robot=True, dt=dt, identifier=mantrap.constants.ID_EGO)
        else:
            assert ego_position is None
            self._ego = None

        self._ados = []  # type: typing.List[mantrap.agents.base.DTAgent]
        self._ado_ids = []  # type: typing.List[str]

        # Dictionary of environment parameters.
        self._env_params = dict()
        self._env_params[mantrap.constants.PK_X_AXIS] = x_axis
        self._env_params[mantrap.constants.PK_Y_AXIS] = y_axis
        self._env_params[mantrap.constants.PK_CONFIG] = config_name
        self._dt = dt
        self._time = time

        # Perform sanity check for environment and agents.
        assert self.sanity_check()

    ###########################################################################
    # Simulation step #########################################################
    ###########################################################################
    def step(self, ego_action: torch.Tensor) -> typing.Tuple[torch.Tensor, typing.Union[torch.Tensor, None]]:
        """Run environment step (time-step = dt).

        Attention: This method changes the states of all environment agents, by executing the ego action
        and sample from the conditioned ado positional distribution.

        :param ego_action: planned ego control input for current time step (2).
        :returns: ado_states (num_ados, num_modes, 1, 5), ego_next_state (5) in next time step.
        """
        assert mantrap.utility.shaping.check_ego_action(ego_action)
        self._time = self._time + self.dt

        # Unroll future ego trajectory, which is surely deterministic and certain due to the deterministic dynamics
        # assumption. Update ego based on the first action of the input ego policy.
        self._ego.update(ego_action, dt=self.dt)
        logging.debug(f"env {self.log_name} step @t={self.time} [ego]: action={ego_action.tolist()}")
        logging.debug(f"env {self.log_name} step @t={self.time} [ego]: state={self.ego.state.tolist()}")

        # Update the ados by sampling one sample from the distribution. Since only the next position is known,
        # but we dont know whether this action is feasible, e.g. with respect to the agents velocity bounds,
        # the inverse update function is used, which basically computes the action between subsequent states
        # and then updates the agent again, by executing (the feasible part of) this action.
        # Since we assume single integrator dynamics and the current state of each ado is known (and therefore
        # deterministic) the velocity just can be computed by deriving the difference of the sample and the
        # current position of each ado (multiplied by 1/dt).
        ado_states = torch.zeros((self.num_ados, 5))
        ado_samples = self.sample_w_controls(ego_controls=ego_action.view(1, 2), num_samples=1)
        for m_ado, ado_id in enumerate(self.ado_ids):
            ado_m_position = ado_samples[m_ado, 0, 1, 0, :]
            ado_m_velocity = (ado_m_position - self.ados[m_ado].position) / self.dt
            ado_m_state = torch.cat((ado_m_position, ado_m_velocity))

            self._ados[m_ado].update_inverse(state_next=ado_m_state, dt=self.dt)
            ado_states[m_ado, :] = self._ados[m_ado].state_with_time
            logging.debug(f"[ado_{ado_id}] updated => state={ado_states[m_ado].tolist()}")

        # Detach agents from graph in order to keep independence between subsequent runs. Afterwards perform sanity
        # check for environment and agents.
        self.detach()
        assert self.sanity_check()

        assert mantrap.utility.shaping.check_ado_states(ado_states, num_ados=self.num_ados, enforce_temporal=True)
        return ado_states.detach(), self.ego.state_with_time.detach()  # otherwise no scene independence (!)

    def step_reset(self, ego_next: typing.Union[torch.Tensor, None], ado_next: typing.Union[torch.Tensor, None]):
        """Run environment step (time-step = dt). Instead of predicting the behaviour of every agent in the scene, it
        is given as an input and the agents are merely updated. All the ghosts (modes of an ado) will collapse to the
        same given state, since the update is deterministic.

        :param ego_next: ego state for next time step (5).
        :param ado_next: ado states for next time step (num_ados, num_modes, 1, 5).
        """
        self._time = self._time + self.dt

        # Reset ego agent (if there is an ego in the scene), otherwise just do not reset it.
        if ego_next is not None:
            assert mantrap.utility.shaping.check_ego_state(ego_next, enforce_temporal=True)
            self._ego.reset(state=ego_next, history=None)  # new state is appended

        # Reset ado agents, if `ado_states_next` is None just do not reset them. When resetting with `history=None`
        # the new state is appended automatically (see mantrap.agents).
        if ado_next is not None:
            assert mantrap.utility.shaping.check_ado_states(ado_next, self.num_ados, enforce_temporal=True)
            for m_ado in range(self.num_ados):
                self._ados[m_ado].reset(ado_next[m_ado, :], history=None)

        # Detach agents from graph in order to keep independence between subsequent runs. Afterwards perform sanity
        # check for environment and agents.
        self.detach()
        assert self.sanity_check()

    ###########################################################################
    # Prediction - Samples ####################################################
    ###########################################################################
    def sample_w_controls(self, ego_controls: torch.Tensor, num_samples: int = 1, expand: bool = False
                          ) -> typing.Union[torch.Tensor, None]:
        """Predict the ado path samples based conditioned on robot controls.

        :param ego_controls: ego control input (pred_horizon, 2).
        :param num_samples: number of samples to return.
        :param expand: expand ado positions outputs to full state trajectories.
        :return: predicted ado paths (num_ados, num_samples, pred_horizon+1, 2/5).
                 if no ado in scene, return None instead.
        """
        assert mantrap.utility.shaping.check_ego_controls(ego_controls)
        assert self.sanity_check(check_ego=True)

        ego_trajectory = self.ego.unroll_trajectory(controls=ego_controls, dt=self.dt)
        return self.sample_w_trajectory(ego_trajectory, num_samples=num_samples, expand=expand)

    def sample_w_trajectory(self, ego_trajectory: torch.Tensor, num_samples: int = 1, expand: bool = False
                            ) -> typing.Union[torch.Tensor, None]:
        """Predict the ado path samples based conditioned on robot trajectory.

        :param ego_trajectory: ego trajectory (pred_horizon + 1, 5).
        :param num_samples: number of samples to return.
        :param expand: expand ado positions outputs to full state trajectories.
        :return: predicted ado paths (num_ados, num_samples, pred_horizon+1, 2/5).
                 if no ado in scene, return None instead.
        """
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, pos_and_vel_only=True)
        assert self.sanity_check(check_ego=True)
        t_horizon = ego_trajectory.shape[0] - 1

        # If no ado agent is in the scene, then return None.
        if self.num_ados == 0:
            return None

        # Otherwise predict the conditioned distribution and draw samples from them.
        dist_dict = self.compute_distributions(ego_trajectory=ego_trajectory)
        samples = torch.stack([dist_dict[ado_id].sample((num_samples, )) for ado_id in self.ado_ids])
        if expand:
            samples = self.expand_ado_trajectories(ado_trajectories=samples)

        assert mantrap.utility.shaping.check_ado_samples(samples, t_horizon + 1, self.num_ados, num_samples)
        return samples

    def sample_wo_ego(self, t_horizon: int, num_samples: int = 1, expand: bool = False
                      ) -> typing.Union[torch.Tensor, None]:
        """Predict the unconditioned ado path samples (i.e. if no robot would be in the scene).

        :param t_horizon: prediction horizon, number of discrete time-steps.
        :param num_samples: number of samples to return.
        :param expand: expand ado positions outputs to full state trajectories.
        :return: predicted ado paths (num_ados, num_samples, pred_horizon+1, 2/5).
                 if no ado in scene, return None instead.
        """
        assert t_horizon > 0
        assert self.sanity_check(check_ego=False)

        # If no ado agent is in the scene, then return None.
        if self.num_ados == 0:
            return None

        # Otherwise predict the un_conditioned distribution and draw samples from them.
        dist_dict = self.compute_distributions_wo_ego(t_horizon=t_horizon)
        samples = torch.stack([dist_dict[ado_id].sample((num_samples,)) for ado_id in self.ado_ids])
        if expand:
            samples = self.expand_ado_trajectories(ado_trajectories=samples)

        assert mantrap.utility.shaping.check_ado_samples(samples, t_horizon + 1, self.num_ados, num_samples)
        return samples

    ###########################################################################
    # Prediction - Means ######################################################
    ###########################################################################
    def predict_w_controls(self, ego_controls: torch.Tensor, expand: bool = False
                           ) -> typing.Union[torch.Tensor, None]:
        """Predict the ado path distribution means based conditioned on robot controls.

        :param ego_controls: ego control input (pred_horizon, 2).
        :param expand: expand ado positions outputs to full state trajectories.
        :return: predicted ado paths (num_ados, num_samples, pred_horizon+1, 2/5).
                 if no ado in scene, return None instead.
        """
        assert mantrap.utility.shaping.check_ego_controls(ego_controls)
        assert self.sanity_check(check_ego=True)

        ego_trajectory = self.ego.unroll_trajectory(controls=ego_controls, dt=self.dt)
        return self.predict_w_trajectory(ego_trajectory, expand=expand)

    def predict_w_trajectory(self, ego_trajectory: torch.Tensor, expand: bool = False
                             ) -> typing.Union[torch.Tensor, None]:
        """Predict the ado path samples based conditioned on robot trajectory.

        :param ego_trajectory: ego trajectory (pred_horizon + 1, 5).
        :param expand: expand ado positions outputs to full state trajectories.
        :return: predicted ado paths (num_ados, num_samples, pred_horizon+1, 2/5).
                 if no ado in scene, return None instead.
        """
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, pos_and_vel_only=True)
        assert self.sanity_check(check_ego=True)
        t_horizon = ego_trajectory.shape[0] - 1

        # If no ado agent is in the scene, then return None.
        if self.num_ados == 0:
            return None

        # Otherwise predict the un_conditioned distribution and draw samples from them.
        dist_dict = self.compute_distributions(ego_trajectory=ego_trajectory)
        means = torch.stack([dist_dict[ado_id].mean for ado_id in self.ado_ids])
        if len(means.shape) == 3:
            means = means.unsqueeze(dim=1)
        if expand:
            means = self.expand_ado_trajectories(ado_trajectories=means)

        assert mantrap.utility.shaping.check_ado_trajectories(means, t_horizon=t_horizon + 1, ados=self.num_ados)
        return means

    def predict_wo_ego(self, t_horizon: int, expand: bool = False
                       ) -> typing.Union[torch.Tensor, None]:
        """Predict the unconditioned ado path distribution means (i.e. if no robot would be in the scene).

        :param t_horizon: prediction horizon, number of discrete time-steps.
        :param expand: expand ado positions outputs to full state trajectories.
        :return: predicted ado paths (num_ados, num_samples, pred_horizon+1, 2/5).
                 if no ado in scene, return None instead.
        """
        assert t_horizon > 0
        assert self.sanity_check(check_ego=False)

        # If no ado agent is in the scene, then return None.
        if self.num_ados == 0:
            return None

        # Otherwise predict the un_conditioned distribution and draw samples from them.
        dist_dict = self.compute_distributions_wo_ego(t_horizon=t_horizon)
        means = torch.stack([dist_dict[ado_id].mean for ado_id in self.ado_ids])
        if expand:
            means = self.expand_ado_trajectories(ado_trajectories=means)

        assert mantrap.utility.shaping.check_ado_trajectories(means, t_horizon=t_horizon + 1, ados=self.num_ados)
        return means

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

        To represent pedestrians (ados) single integrator dynamics are best suitable due to their dynamic, reactive
        and fast changing nature.

        While the ego is added to the environment during initialization, the ado agents have to be added afterwards,
        individually. To do so initialize single integrator agent using its state vectors, namely position, velocity
        and its state history. The ado id, color and other parameters can either be passed using the ado_kwargs
        option or are created automatically during the agent's initialization.

        After initialization check whether the given states are valid, i.e. do  not pass the internal environment
        bounds, e.g. that they are in the given 2D space the environment is defined in.

        :param position: ado initial position (2D).
        :param velocity: ado initial velocity (2D).
        :param history: ado state history (if None then just current state as history).
        :param ado_kwargs: addition kwargs for ado initialization.
        """
        ado = mantrap.agents.IntegratorDTAgent(position, velocity=velocity, history=history, dt=self.dt, **ado_kwargs)
        self._ado_ids.append(ado.id)
        self._ados.append(ado)

        # Append ado to internal list of ados and rebuilt the graph (could be also extended but small computational
        # to actually rebuild it).
        assert self.axes[0][0] <= ado.position[0] <= self.axes[0][1]
        assert self.axes[1][0] <= ado.position[1] <= self.axes[1][1]

        # Perform sanity check for environment and agents.
        assert self.sanity_check()
        return ado

    def agent_by_id(self, agent_id: str) -> typing.Union[mantrap.agents.base.DTAgent, None]:
        """Return an agent object by its id, including the ego agent.

        :param agent_id: identifier of agent to return.
        """
        if agent_id == mantrap.constants.ID_EGO:
            return self.ego
        else:
            return self.ados[self.index_ado_id(ado_id=agent_id)]

    ###########################################################################
    # Scene State #############################################################
    ###########################################################################
    def states(self) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Return current states of ego and ado agents in the scene. Since the current state is known for every
        ado the states are deterministic and uni-modal. States are returned as vector including temporal dimension.

        :returns: ego state vector including temporal dimension (5).
        :returns: ado state vectors including temporal dimension (num_ados, 5).
        """
        ado_states = torch.zeros((self.num_ados, 5))
        for m_ado, ado in enumerate(self.ados):
            ado_states[m_ado, :] = ado.state_with_time
        ego_state = self.ego.state_with_time if self.ego is not None else None

        if ego_state is not None:
            assert mantrap.utility.shaping.check_ego_state(ego_state, enforce_temporal=True)
        assert mantrap.utility.shaping.check_ado_states(ado_states, enforce_temporal=True, num_ados=self.num_ados)
        return ego_state, ado_states

    def expand_ado_trajectories(self, ado_trajectories: torch.Tensor) -> torch.Tensor:
        assert mantrap.utility.shaping.check_ado_trajectories(ado_trajectories)
        num_ados, num_samples, t_horizon, _ = ado_trajectories.shape
        trajectories_full = torch.zeros((num_ados, num_samples, t_horizon, 5))
        for m_ado, ado in enumerate(self.ados):
            for m_sample in range(num_samples):
                trajectories_full[m_ado, m_sample, :, :] = \
                    ado.expand_trajectory(ado_trajectories[m_ado, m_sample, :, 0:2], dt=self.dt)
        return trajectories_full

    ###########################################################################
    # Simulation graph ########################################################
    ###########################################################################
    def compute_distributions(self, ego_trajectory: torch.Tensor, **kwargs
                              ) -> typing.Dict[str, torch.distributions.Distribution]:
        """Build a dictionary of positional distributions for every ado as it would be with the presence
        of a robot in the scene.

        Build the graph conditioned on some `ego_trajectory`, which is assumed to be fix while the ados in the scene
        behave accordingly, i.e. in reaction to the ego's trajectory. For the sake of differentiability using the
        by building the dictionary using PyTorch, a computational graph is built in the background which can later
        be used for automatically differentiate between its inputs and outputs.

        :param ego_trajectory: ego's trajectory (t_horizon, 5).
        :kwargs: additional graph building arguments.
        :return: dictionary over every state of every agent in the scene for t in [0, t_horizon].
        """
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, pos_and_vel_only=True)
        assert self.ego is not None
        dist_dict = self._compute_distributions(ego_trajectory=ego_trajectory, **kwargs)
        assert self.check_distribution(dist_dict, t_horizon=ego_trajectory.shape[0] - 1)
        return dist_dict

    @abc.abstractmethod
    def _compute_distributions(self, ego_trajectory: torch.Tensor, **kwargs
                               ) -> typing.Dict[str, torch.distributions.Distribution]:
        """Build a connected graph based on the ego's trajectory.

        The graph should span over the time-horizon of the length of the ego's trajectory and contain the
        positional distribution of every ado in the scene as well as the ego's states itself. When
        possible the graph should be differentiable, such that finding some gradient between the outputted ado
        states and the inputted ego trajectory is determinable.

        :param ego_trajectory: ego's trajectory (t_horizon, 5).
        :return: ado_id-keyed positional distribution dictionary for times [0, t_horizon].
        """
        raise NotImplementedError

    def compute_distributions_wo_ego(self, t_horizon: int, **kwargs
                                     ) -> typing.Dict[str, torch.distributions.Distribution]:
        """Build a dictionary of positional distributions for every ado as it would be without the presence
        of a robot in the scene.

        :param t_horizon: number of prediction time-steps.
        :kwargs: additional graph building arguments.
        :return: ado_id-keyed positional distribution dictionary for times [0, t_horizon].
        """
        assert t_horizon > 0
        dist_dict = self._compute_distributions_wo_ego(t_horizon, **kwargs)
        assert self.check_distribution(dist_dict, t_horizon=t_horizon)
        return dist_dict

    @abc.abstractmethod
    def _compute_distributions_wo_ego(self, t_horizon: int, **kwargs
                                      ) -> typing.Dict[str, torch.distributions.Distribution]:
        """Build a connected graph over `t_horizon` time-steps for ados only (exclude robot).

        The graph should span over the time-horizon of the inputted number of time-steps and contain the
        positional distribution of every ado in the scene, stored in an `ado_id` key-ed dictionary. When
        possible the graph should be differentiable, such that finding some gradient between the outputted ado
        states and the inputted ego trajectory is determinable.

        :param t_horizon: number of prediction time-steps.
        :return: dictionary over every state of every ado in the scene for t in [0, t_horizon].
        """
        raise NotImplementedError

    def check_distribution(self, distribution: typing.Dict[str, torch.distributions.Distribution], t_horizon: int):
        """Check the distribution dictionary for correctness."""
        assert all([ado_id in distribution.keys() for ado_id in self.ado_ids])
        assert all([distribution[ado_id].mean.shape[0] == t_horizon + 1 for ado_id in self.ado_ids])
        assert all([distribution[ado_id].mean.shape[-1] == 2 for ado_id in self.ado_ids])
        return True

    def detach(self):
        """Detach all internal agents (ego and all ados) from computation graph. This is sometimes required to
        completely separate subsequent computations in PyTorch."""
        self._ego.detach()
        for m in range(self.num_ados):
            self.ados[m].detach()

    ###########################################################################
    # Operators ###############################################################
    ###########################################################################
    def copy(self, env_type: 'GraphBasedEnvironment'.__class__ = None) -> 'GraphBasedEnvironment':
        """Create copy of environment.

        However just using deepcopy is not supported for tensors that are not detached from the PyTorch
        computation graph. Therefore re-initialize the objects such as the agents in the environment and reset
        their state to the internal current state.

        While copying the environment-type can be defined by the user, which is possible due to standardized
        class interface of every environment-type. When no environment is defined, the default environment
        will be used which is the type of the executing class object.
        """
        env_type = env_type if env_type is not None else self.__class__

        with torch.no_grad():
            # Create environment copy of internal class, pass environment parameters such as the forward integration
            # time-step and initialize ego agent.
            ego_type = None
            ego_kwargs = None
            if self.ego is not None:
                ego_type = self.ego.__class__
                position = self.ego.position
                velocity = self.ego.velocity
                history = self.ego.history
                ego_kwargs = {"ego_position": position, "ego_velocity": velocity, "ego_history": history}

            env_copy = env_type(**ego_kwargs, ego_type=ego_type, dt=self.dt, time=self.time, **self._env_params)

            # Add internal ado agents to newly created environment.
            for ado in self.ados:
                env_copy.add_ado(position=ado.position, velocity=ado.velocity, history=ado.history,
                                 time=self.time, color=ado.color, identifier=ado.id)

        assert self.same_initial_conditions(other=env_copy)
        assert env_copy.sanity_check()
        return env_copy

    def same_initial_conditions(self, other: 'GraphBasedEnvironment'):
        """Similar to __eq__() function, but not enforcing parameters of environment to be completely equivalent,
        merely enforcing the initial conditions to be equal, such as states of agents in scene. Hence, all prediction
        depending parameters, dont have to be equal.

        :param other: comparable environment object.
        """
        assert self.dt == other.dt
        assert self.num_ados == other.num_ados
        assert self.ego.__eq__(other.ego, check_class=True)

        for ado in self.ados:
            other_ado_index = other.index_ado_id(ado_id=ado.id)
            assert ado.__eq__(other.ados[other_ado_index], check_class=False)

        return True

    ###########################################################################
    # Sanity Check ############################################################
    ###########################################################################
    def sanity_check(self, check_ego: bool = False) -> bool:
        """Check for the sanity of the scene and agents.

        For the environment to be "sane" general environment properties should hold as well as all agents living
        in the environment should be "sane" as well. For example it is ensured that the number and order of `ado_ids`
        is equal to the list of `ados` itself.
        """
        assert self.num_ados == len(self.ado_ids)
        assert all([self.ados[m_ado].id == self.ado_ids[m_ado] for m_ado in range(self.num_ados)])

        # Check sanity of all agents in the scene.
        if check_ego:
            assert self.ego is not None
            assert self.ego.is_robot
            assert self.ego.sanity_check()
        for ado in self.ados:
            assert ado.sanity_check()

        return True

    ###########################################################################
    # Visualization ###########################################################
    ###########################################################################
    def visualize_prediction(self, ego_trajectory: torch.Tensor, **vis_kwargs):
        """Visualize the predictions for the scene based on the given ego trajectory."""
        from mantrap.visualization import visualize_prediction
        from mantrap.visualization.atomics import output_format
        assert mantrap.utility.shaping.check_ego_trajectory(x=ego_trajectory)
        t_horizon = ego_trajectory.shape[0] - 1

        # Predict the ado behaviour conditioned on the given ego trajectory.
        num_samples = mantrap.constants.VISUALIZATION_SAMPLES
        ado_samples = self.sample_w_trajectory(ego_trajectory=ego_trajectory, num_samples=num_samples)
        ado_samples_wo = self.sample_wo_ego(t_horizon=t_horizon, num_samples=num_samples)

        return visualize_prediction(
            ego_planned=ego_trajectory,
            ado_planned=ado_samples,
            ado_planned_wo=ado_samples_wo,
            env=self,
            file_path=output_format(name=f"{self.log_name}_prediction"),
            **vis_kwargs
        )

    def visualize_prediction_wo_ego(self, t_horizon: int, **vis_kwargs):
        """Visualize the predictions for the scene based on the given ego trajectory."""
        from mantrap.visualization import visualize_prediction
        from mantrap.visualization.atomics import output_format

        # Predict the ado behaviour conditioned on the given ego trajectory.
        num_samples = mantrap.constants.VISUALIZATION_SAMPLES
        ado_samples_wo = self.sample_wo_ego(t_horizon=t_horizon, num_samples=num_samples)

        output_path = output_format(name=f"{self.log_name}_prediction_wo_ego")
        return visualize_prediction(ado_planned_wo=ado_samples_wo, env=self, file_path=output_path, **vis_kwargs)

    ###########################################################################
    # Ado properties ##########################################################
    ###########################################################################
    @property
    def ados(self) -> typing.List[mantrap.agents.base.DTAgent]:
        return self._ados

    @property
    def ado_colors(self) -> typing.List[np.ndarray]:
        return [ado.color for ado in self.ados]

    @property
    def ado_ids(self) -> typing.List[str]:
        return self._ado_ids

    @property
    def num_ados(self) -> int:
        return len(self.ados)

    def index_ado_id(self, ado_id: str) -> int:
        return self.ado_ids.index(ado_id)

    ###########################################################################
    # Ego properties ##########################################################
    ###########################################################################
    @property
    def ego(self) -> typing.Union[mantrap.agents.base.DTAgent, None]:
        return self._ego

    ###########################################################################
    # Simulation parameters ###################################################
    ###########################################################################
    @property
    def dt(self) -> float:
        return self._dt

    @property
    def time(self) -> float:
        return round(self._time, 2)

    @property
    def axes(self) -> typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]]:
        return self._env_params[mantrap.constants.PK_X_AXIS], self._env_params[mantrap.constants.PK_Y_AXIS]

    @property
    def x_axis(self) -> typing.Tuple[float, float]:
        return self._env_params[mantrap.constants.PK_X_AXIS]

    @property
    def y_axis(self) -> typing.Tuple[float, float]:
        return self._env_params[mantrap.constants.PK_Y_AXIS]

    @property
    def config_name(self) -> str:
        return self._env_params[mantrap.constants.PK_CONFIG]

    @property
    def log_name(self) -> str:
        return self.name + "_" + self.config_name

    ###########################################################################
    # Simulation properties ###################################################
    ###########################################################################
    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def num_modes(self) -> int:
        raise NotImplementedError

    @property
    def is_differentiable_wrt_ego(self) -> bool:
        raise NotImplementedError
