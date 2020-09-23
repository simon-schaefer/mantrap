import json
import os
import sys
import typing

import numpy as np
import pandas as pd
import torch
import torch.distributions

import mantrap.agents
import mantrap.constants
import mantrap.utility.io
import mantrap.utility.maths
import mantrap.utility.shaping

from .base import GraphBasedEnvironment


class Trajectron(GraphBasedEnvironment):
    """Trajectron-based environment model (B. Ivanovic, T. Salzmann, M. Pavone).

    The Trajectron model requires to get some robot position. Therefore, in order to minimize the
    impact of the ego robot on the trajectories (since the prediction should be not conditioned on the robot)
    some pseudo trajectory is used, which is very far distant from the actual scene.

    Within the trajectory optimisation the ado's trajectories conditioned on the robot's planned motion are
    compared with their trajectories without taking any robot into account. So when both the conditioned and
    un-conditioned model for these predictions would be used, and they would be behavioral different, it would
    lead to some base difference (even if there is no robot affecting some ado at all) which might be larger in
    scale than the difference the conditioning on the robot makes. Then minimizing the difference would miss the
    goal of minimizing interaction.
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
        # Load trajectron configuration dictionary and check against inputs.
        config_file_path = mantrap.utility.io.build_os_path("mantrap/trajectron.json")
        self._config = self.load_and_check_configuration(config_path=config_file_path)
        assert dt == self.config["dt"]

        # Initialize environment mother class.
        super(Trajectron, self).__init__(ego_position, ego_velocity, ego_history, ego_type, dt=dt, **env_kwargs)

        # For prediction un-conditioned on the ego (`sample_wo_ego()`) we need a pseudo-ego trajectory, since the
        # input dimensions for the trajectron have to stay the same.
        pseudo_ego_position = torch.tensor([self.axes[0][0], self.axes[1][0]])
        self._pseudo_ego = mantrap.agents.DoubleIntegratorDTAgent(pseudo_ego_position,
                                                                  velocity=torch.zeros(2),
                                                                  is_robot=True,
                                                                  identifier=mantrap.constants.ID_EGO)

        # Create trajectron torch model with loaded configuration.
        from model.online.online_trajectron import OnlineTrajectron
        from model.model_registrar import ModelRegistrar
        model_registrar = ModelRegistrar(model_dir=self.config["trajectron_model_path"], device="cpu")
        model_registrar.load_models(iter_num=self.config["trajectron_model_iteration"])
        self.trajectron = OnlineTrajectron(model_registrar, hyperparams=self.config, device="cpu")

        # Create default trajectron scene. The duration of the scene is not known a priori, however a large value
        # allows to simulate for a long time horizon later on.
        self._gt_scene, self._gt_env = self.create_env_and_scene()

        # Add robot to the scene as a first node.
        self._online_env = None
        self._add_agent_to_graph(agent=self.ego if self.ego is not None else self._pseudo_ego)

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

        Next to the internal ado representation of each ado within this class, the Trajectron model has an own
        graph, in which the ado has to be introduced as a node during initialization. Also Trajectron model is
        trained to predict accurately, iff the agent has some history > 1, therefore if no history is given
        build a custom history by stacking the given (position, velocity) state over multiple time-steps. If
        zero history should be enforced, pass a non None history argument.

        :param position: ado initial position (2D).
        :param velocity: ado initial velocity (2D).
        :param history: ado state history (if None then just stacked current state).
        :param ado_kwargs: addition kwargs for ado initialization.
        """
        # When being queried with an agent's history length of one, the Trajectron will always predict standing
        # still instead of assuming a constant velocity (or likewise). However when the user inputs just one
        # position and one velocity, rather than a full state history, continuing on this path clearly is the
        # user's intention, therefore stack several copies of this input state (position, velocity, time) in order
        # to create a history the Trajectron is used to deal with, while giving the incentive to predict a simple
        # "constant" continuation of this state as most likely prediction.
        if history is None or history.shape[0] == 1:
            position, velocity = position.float(), velocity.float()
            history = torch.stack([torch.cat(
                (position + velocity * self.dt * t, velocity, torch.ones(1) * self.time + self.dt * t))
                for t in range(-mantrap.constants.TRAJECTRON_DEFAULT_HISTORY_LENGTH, 1)
            ])

        ado = super(Trajectron, self).add_ado(position, velocity=velocity, history=history, **ado_kwargs)
        # Add a ado to Trajectron neural network model using reference ghost.
        self._add_agent_to_graph(agent=ado)
        return ado

    def _add_agent_to_graph(self, agent: mantrap.agents.base.DTAgent):
        """Add an internal agent to the Trajectron scene/environment representation.

        :param agent: agent object to add (either ado, ego or pseudo_ego).
        """
        from data import Node
        is_robot = agent.is_robot

        # In Trajectron each node has a certain type, which is either robot or pedestrian, an id and
        # state data. Enforce the Trajectron id to the internal ids format, to be able to query the
        # results later on.
        agent_history = agent.history
        acc_history = agent.compute_acceleration(agent_history, dt=self.dt)

        node_data = self._create_node_data(state_history=agent_history, accelerations=acc_history)
        node_tye = self._gt_env.NodeType.PEDESTRIAN if not is_robot else self._gt_env.NodeType.ROBOT
        node = Node(node_type=node_tye, node_id=agent.id, data=node_data, is_robot=is_robot)
        if is_robot:
            self._gt_scene.robot = node
        self._gt_scene.nodes.append(node)

        # Re-Create online environment with recently appended node.
        self._online_env = self.create_online_env(env=self._gt_env, scene=self._gt_scene)

    @staticmethod
    def agent_id_from_node(node: str) -> str:
        """In Trajectron nodes have an identifier structure as follows "node_type/node_id". As initialized the node_id
        is identical to the internal node_id while is node type is e.g. "ROBOT" or "PEDESTRIAN". However it is not
        assumed that the node_type has to be robot or pedestrian, since it does not change the structure. """
        return node.__str__().split("/")[1]

    def agent_by_id(self, agent_id: str) -> typing.Union[mantrap.agents.base.DTAgent, None]:
        agent = super(Trajectron, self).agent_by_id(agent_id=agent_id)
        if agent is None:
            return self._pseudo_ego
        return agent

    ###########################################################################
    # Simulation Graph ########################################################
    ###########################################################################
    def _compute_distributions(self, ego_trajectory: torch.Tensor, vel_dist: bool = True, **kwargs
                               ) -> typing.Dict[str, torch.distributions.Distribution]:
        """Build a connected graph based on the ego's trajectory.

        The graph should span over the time-horizon of the length of the ego's trajectory and contain the
        velocity distribution of every ado in the scene as well as the ego's states itself. When
        possible the graph should be differentiable, such that finding some gradient between the outputted ado
        states and the inputted ego trajectory is determinable.

        The Trajectron directly predicts the full velocity distribution for each ado, as a GMM (Gaussian
        Mixture Model) with 25 modes. The GMM is a multi-nominal distribution with weight parameters pi_i,
        i.e. we have

        .. math:: z_1, ..., z_n \\sim Mult_g(1, \\pi_1, ..., \\pi_g)

        with z_i denoting the unobservable component-indicator vector, showing to which out of g clusters a drawn
        sample belongs to (https://books.google.de/books?id=-0mfDwAAQBAJ&pg=PA18&lpg=PA18&dq=Log+Mixing+Proportions).

        mus.shape: (num_ados = 1, 1, t_horizon, num_modes, 2)
        log_pis.shape: (num_ados = 1, 1, t_horizon, num_modes)

        :param ego_trajectory: ego's trajectory (t_horizon, 5).
        :param vel_dist: return velocity (True) or positional distribution (False).
        :return: ado_id-keyed velocity distribution dictionary for times [0, t_horizon].
        """
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, pos_and_vel_only=True)
        assert self.num_ados > 0  # trajectron conditioned on ados and ego, so both must be in the scene (!)
        t_horizon = ego_trajectory.shape[0] - 1

        # Create the buffer of agent state histories to pass to environment. As Trajectron is called several
        # times in each environment step, we do not want it to store agent updates internally, but rather
        # pass it the full agent histories.
        pos_dicts = []
        for t in range(-5, 0):
            node_state_dict = {}
            for node in self._gt_scene.nodes:
                if node.id == mantrap.constants.ID_EGO:
                    node_state_dict[node] = self.agent_by_id(node.id).position.detach()
                else:
                    node_state_dict[node] = self.agent_by_id(node.id).history[t, 0:2].detach()
            pos_dicts.append(node_state_dict)

        # Trajectron requires the ego trajectory to consist of (pos, velocity, acceleration). So compute
        # the accelerations using numerical differentiation. Although the pseudo-ego is used here, it is
        # of the same agent type as the actual ego (which might not be defined).
        accelerations = self._pseudo_ego.compute_acceleration(ego_trajectory, dt=self.dt)
        trajectory_w_acc = torch.cat((ego_trajectory[:, 0:4], accelerations), dim=1)

        # Core trajectron prediction call (returning velocity distribution !).
        trajectron_dist_dict, _ = self.trajectron.forward(
            init_env=self._online_env,
            init_timestep=0,
            pos_dicts=pos_dicts,
            num_predicted_timesteps=t_horizon,
            num_samples=1,
            full_dist=True,
            vel_dist=vel_dist,
            robot_present_and_future=trajectory_w_acc
        )

        # Build ado-wise dictionary distribution of probability distributions. The Trajectron distribution is a
        # dictionary mapping the GenTrajectron tag ("{class_id}/{id}") to a distribution object, defined in gmm2d.py
        # in its code base (GMM with n = 25 modes).
        dist_dict = {}
        m = self.num_modes
        _, ado_states = self.states()
        for node, dist in trajectron_dist_dict.items():
            # Re-Map the node-id to the agent tags using within this project during initialization
            # (enforced to be identical except of type-tag during initialization).
            ado_id = self.agent_id_from_node(node)
            i_ado = self.index_ado_id(ado_id=ado_id)

            # Shift (relative) distribution to initial absolute position (if positional distribution).
            mus = dist.mus.view(t_horizon, m, 2)  # t_horizon, num_modes, num_dims = 2 (= x, y)
            if not vel_dist:  # positional distribution
                mus += ado_states[i_ado, 0:2]

            # Convert the distribution into the project-custom definition of a GMM, since some properties
            # as e.g. mean are not defined in gmm2d.py and since another shape format is used.
            log_sigmas = dist.log_sigmas.view(t_horizon, m, 2)
            corrs = dist.corrs.view(t_horizon, m)
            log_pis = dist.log_pis.view(t_horizon, m)
            distribution = mantrap.utility.maths.VGMM2D(mus=mus, log_pis=log_pis, log_sigmas=log_sigmas, corrs=corrs)
            dist_dict[ado_id] = distribution

        return dist_dict

    def _compute_distributions_wo_ego(self, t_horizon: int, vel_dist: bool = True, **kwargs
                                      ) -> typing.Dict[str, torch.distributions.Distribution]:
        """Build a connected graph over `t_horizon` time-steps for ados only.

        The graph should span over the time-horizon of the inputted number of time-steps and contain the state
        (position, velocity) and "controls" of every ghost in the scene as well as the ego's states itself. When
        possible the graph should be differentiable, such that finding some gradient between the outputted ado
        states and the inputted ego trajectory is determinable.

        The Trajectron model is conditioned on some ego trajectory. Therefore in order to "simulate" the behaviour
        of the agents in the scene if no ego would be there, a "pseudo"-ego-trajectory is built, by shifting it
        to the borders of the environment and having nearly zero velocity.

        :param t_horizon: number of prediction time-steps.
        :param vel_dist: return velocity (True) or positional distribution (False).
        :return: dictionary over every state of every ado in the scene for t in [0, t_horizon].
        """
        pseudo_trajectory = self._pseudo_ego.unroll_trajectory(torch.zeros((t_horizon, 2)), dt=self.dt)
        return self._compute_distributions(pseudo_trajectory, vel_dist=vel_dist, **kwargs)

    def detach(self):
        """Detaching the whole graph (which is the whole neural network) might be hard. Therefore just rebuilt it
        from scratch completely, using the most up-to-date states of the agents. """
        super(Trajectron, self).detach()

        # Reset internal scene representation.
        self._gt_scene.nodes = []
        self._gt_scene.robot = None

        # Add all agents to the scene again.
        self._add_agent_to_graph(agent=self.ego if self.ego is not None else self._pseudo_ego)
        for ado in self.ados:
            self._add_agent_to_graph(agent=ado)

    ###########################################################################
    # GenTrajectron ###########################################################
    ###########################################################################
    @staticmethod
    def import_modules():
        if Trajectron.module_os_path() not in sys.path:
            sys.path.insert(0, Trajectron.module_os_path())

    def create_env_and_scene(self):
        from data import Environment, Scene
        scene = Scene(timesteps=100, map=None, dt=self.dt)
        env = Environment(**self.config["env_params"])
        env.attention_radius = {
            (env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): self.config["attention_radius_pp"],
            (env.NodeType.ROBOT, env.NodeType.ROBOT): 0.0,
            (env.NodeType.ROBOT, env.NodeType.PEDESTRIAN): self.config["attention_radius_pr"],
            (env.NodeType.PEDESTRIAN, env.NodeType.ROBOT): self.config["attention_radius_rp"],
        }
        return scene, env

    def create_online_env(self, env, scene, init_time_step: int = 0):
        from data import Environment, Scene

        # Update environment with the current scene (by replacing the old scene).
        env.scenes = [scene]  # only one scene (online !)

        # Creating online scene and initialize environment.
        online_scene = Scene(timesteps=init_time_step + 1, map=scene.map, dt=scene.dt)
        time_steps = np.arange(init_time_step - self.config["maximum_history_length"], init_time_step + 1)
        online_scene.nodes = scene.get_nodes_clipped_at_time(timesteps=time_steps, state=self.config["state"])
        online_scene.robot = scene.robot
        online_scene.calculate_scene_graph(attention_radius=env.attention_radius,
                                           edge_addition_filter=self.config["edge_addition_filter"],
                                           edge_removal_filter=self.config["edge_removal_filter"])
        online_env = Environment(
            node_type_list=env.node_type_list,
            standardization=env.standardization,
            scenes=[online_scene],
            attention_radius=env.attention_radius,
            robot_type=env.robot_type
        )
        return online_env

    ###########################################################################
    # Utility #################################################################
    ###########################################################################
    def load_and_check_configuration(self, config_path: str) -> typing.Dict[str, typing.Any]:
        self.import_modules()
        from argument_parser import args

        # Load configuration files.
        model, iteration = mantrap.constants.TRAJECTRON_MODEL
        trajectron_path = mantrap.utility.io.build_os_path(f"third_party/trajectron_models/{model}")
        config = {"trajectron_model_path": trajectron_path, "trajectron_model_iteration": iteration}
        with open(config_path) as trajectron_config_file:
            config.update(json.load(trajectron_config_file))
        with open(os.path.join(config["trajectron_model_path"], args.conf)) as model_config_file:
            config.update(json.load(model_config_file))

        # Corrections of loaded configuration to make it usable for scene including robots.
        robot_state = {"ROBOT": {"position": ["x", "y"], "velocity": ["x", "y"], "acceleration": ["x", "y"]}}
        config["state"].update(robot_state)
        config["dynamic"].update({"ROBOT": "DoubleIntegrator"})
        config["incl_robot_node"] = True
        return config

    @staticmethod
    def _create_node_data(state_history: torch.Tensor, accelerations: torch.Tensor) -> pd.DataFrame:
        assert mantrap.utility.shaping.check_ego_trajectory(state_history, pos_and_vel_only=True)

        data_dict = {
            ("position", "x"): state_history[:, 0],
            ("position", "y"): state_history[:, 1],
            ("velocity", "x"): state_history[:, 2],
            ("velocity", "y"): state_history[:, 3],
            ("acceleration", "x"): accelerations[:, 0],
            ("acceleration", "y"): accelerations[:, 1]
        }
        data_columns = pd.MultiIndex.from_product([["position", "velocity", "acceleration"], ["x", "y"]])
        return pd.DataFrame(data_dict, columns=data_columns)

    @staticmethod
    def module_os_path() -> str:
        module_path = mantrap.utility.io.build_os_path("third_party/GenTrajectron/code", make_dir=False, free=False)
        assert os.path.isdir(module_path)
        return module_path

    @property
    def config(self) -> typing.Dict[str, typing.Any]:
        return self._config

    ###########################################################################
    # Simulation parameters ###################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "trajectron"

    @property
    def num_modes(self) -> int:
        return 25

    @property
    def is_differentiable_wrt_ego(self) -> bool:
        return True
