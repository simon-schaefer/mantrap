import json
import os
import sys
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch

from mantrap.agents.agent import Agent
from mantrap.agents import IntegratorDTAgent
from mantrap.constants import env_dt_default, env_trajectron_model
from mantrap.environment.environment import GraphBasedEnvironment
from mantrap.utility.io import build_os_path
from mantrap.utility.maths import Derivative2
from mantrap.utility.shaping import check_ego_trajectory


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
        ego_type: Agent.__class__ = None,
        ego_kwargs: Dict[str, Any] = None,
        dt: float = env_dt_default,
        **env_kwargs
    ):
        # Load trajectron configuration dictionary and check against inputs.
        self._config = self.load_and_check_configuration(config_path=build_os_path("config/trajectron.json"))
        assert dt == self.config["dt"]

        # Initialize environment mother class.
        super(Trajectron, self).__init__(ego_type, ego_kwargs, dt=dt, **env_kwargs)

        # For prediction un-conditioned on the ego (`predict_wo_ego()`) we need a pseudo-ego trajectory, since the
        # input dimensions for the trajectron have to stay the same.
        self._pseudo_ego = IntegratorDTAgent(position=torch.tensor(self.axes[0]))

        # Create default trajectron scene. The duration of the scene is not known a priori, however a large value
        # allows to simulate for a long time horizon later on.
        self._gt_scene, self._gt_env = self.create_env_and_scene()

        # Create trajectron torch model with loaded configuration.
        from model.online.online_trajectron import OnlineTrajectron
        from model.model_registrar import ModelRegistrar
        model_registrar = ModelRegistrar(model_dir=self.config["trajectron_model_path"], device="cpu")
        model_registrar.load_models(iter_num=self.config["trajectron_model_iteration"])
        self.trajectron = OnlineTrajectron(model_registrar, hyperparams=self.config, device="cpu")

        # Add robot to the scene as a first node.
        self._online_env = None
        self._add_ego_to_graph(ego=self.ego if self.ego is not None else self._pseudo_ego)

    ###########################################################################
    # Scene ###################################################################
    ###########################################################################
    def add_ado(self, **ado_kwargs):
        """Add a new ado and its mode to the scene.

        For the Trajectron model the multi-modality evolves at the output, not the input. Therefore instead of
        multiple ghosts of the same ado agent just the agent is added to the internal scene graph, as Pedestrian
        node. However the representation of the agents and their modes in the base class intrinsically is multimodal,
        by keeping the different modes as independent agents sharing the same history. Therefore a reference ghost
        is chosen to pass these shared properties to  the Trajectron scene.

        The weight vector for each mode is not constant, rather changing with every new scene and prediction. However,
        in order to compute it for the current scene a forward pass would be required. Since the environment and
        especially the mode's weights can be assumed to not be used without a precedent prediction step, the weights
        are initialized as a not really meaningful uniform distribution for now and then updated during the
        environment's prediction step.
        """
        super(Trajectron, self).add_ado(ado_type=IntegratorDTAgent, **ado_kwargs)

        # Add a ado to Trajectron neural network model using reference ghost.
        ado_id, _ = self.split_ghost_id(ghost_id=self.ghosts[-1].id)
        ado_history = self.ghosts[-1].agent.history
        self._add_ado_to_graph(ado_history=ado_history, ado_id=ado_id)

    def _add_ado_to_graph(self, ado_history: torch.Tensor, ado_id: str):
        from data import Node

        # Add a ado to online environment. Since the model predicts over the full time horizon internally and all
        # modes share the same state history, merely one ghost has to be added to the scene, representing all other
        # modes of the added ado agent.
        node_data = self._create_node_data(state_history=ado_history)
        node = Node(node_type=self._gt_env.NodeType.PEDESTRIAN, node_id=ado_id, data=node_data)
        self._gt_scene.nodes.append(node)

        # Re-Create online environment with recently appended node.
        self._online_env = self.create_online_env(env=self._gt_env, scene=self._gt_scene)

    def _add_ego_to_graph(self, ego: Agent):
        from data import Node

        # Add the ego as robot-type agent to the scene integrating its state history and passing the pre-build
        # ego id as node id. Since the Trajectron model used is conditioned on the robot's movement, each prediction
        # requires this ego. Therefore we have to add a robot to the scene, might it be an actual ego robot or (if
        # no ego is in the scene) some pseudo robot very far away from the other agents in the scene.
        node_data = self._create_node_data(state_history=ego.history)
        node = Node(node_type=self._gt_env.NodeType.ROBOT, node_id=ego.id, data=node_data, is_robot=True)
        self._gt_scene.robot = node
        self._gt_scene.nodes.append(node)

        # Re-Create online environment with recently appended node.
        self._online_env = self.create_online_env(env=self._gt_env, scene=self._gt_scene)

    @staticmethod
    def ado_id_from_node_id(node_id: str) -> str:
        return node_id.split("/")[1]

    ###########################################################################
    # Simulation Graph ########################################################
    ###########################################################################
    def _build_connected_graph(self, t_horizon: int, ego_trajectory: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        assert check_ego_trajectory(ego_trajectory, pos_and_vel_only=True)
        t_horizon = ego_trajectory.shape[0]

        # Transcribe initial states in graph.
        graph = self.write_state_to_graph(ego_trajectory[0], ego_grad=True, ado_grad=False)

        # Predict over full time-horizon at once and write resulting (simulated) ado trajectories to graph. By passing
        # the robot's trajectory the distribution is conditioned on it.
        # Using `full_dist = True` not just the mean of the resulting trajectory but also the covariances are returned.
        dd = Derivative2(horizon=t_horizon, dt=self.dt, velocity=True)
        trajectory_w_acc = torch.cat((ego_trajectory[:, 0:4], dd.compute(ego_trajectory[:, 2:4])), dim=1)
        distribution, _ = self.trajectron.incremental_forward(
            new_inputs_dict=self._gt_scene.get_clipped_pos_dict(0, self.config["state"]),
            prediction_horizon=ego_trajectory.shape[0] - 1,
            num_samples=1,
            full_dist=True,
            robot_present_and_future=trajectory_w_acc
        )

        # Build the state at the current time-step, by using the `environment::states()` method. However, since the
        # state at the current time-step is deterministic, the output vector has to be stretched to the number of
        # modes (while having the same state for modes that are originated from the same ado).
        ado_planned = torch.zeros((self.num_ados, self.num_modes, t_horizon, 5))
        ado_weights = torch.zeros((self.num_ados, self.num_modes))
        _, ado_states = self.states()
        ado_planned[:, :, 0, :] = ado_states.view(self.num_ados, 1, 5).repeat(1, self.num_modes, 1)

        # The obtained distribution is a dictionary mapping nodes to the representing Gaussian Mixture Model (GMM).
        # Most importantly the GMM have a mean (mu) and log-covariance (log_sigma) for every of the 25 modes.
        for node, node_gmm in distribution.items():
            ado_id = self.ado_id_from_node_id(node.__str__())
            m_ado = self.index_ado_id(ado_id)
            ado_planned[m_ado], ado_weights[m_ado] = self.trajectory_from_distribution(
                gmm=node_gmm, num_output_modes=self.num_modes, dt=self.dt, t_horizon=t_horizon, t_start=self.time
            )

        # Update the graph dictionary with the trajectory predictions that have  been derived before.
        for t in range(t_horizon):
            graph[f"ego_{t}_position"] = ego_trajectory[t, 0:2]
            graph[f"ego_{t}_velocity"] = ego_trajectory[t, 2:4]
            for m_ghost, ghost in enumerate(self.ghosts):
                m_ado, m_mode = self.convert_ghost_id(ghost_id=ghost.id)

                graph[f"{ghost.id}_{t}_position"] = ado_planned[m_ado, m_mode, t, 0:2]
                graph[f"{ghost.id}_{t}_velocity"] = ado_planned[m_ado, m_mode, t, 2:4]
                graph[f"{ghost.id}_{t}_control"] = ado_planned[m_ado, m_mode, t, 2:4]  # single integrator ados (!)

                # Adapt weight as determined from prediction.
                self._ado_ghosts[m_ghost].weight = ado_weights[m_ado, m_mode]
        return graph

    def build_connected_graph_wo_ego(self, t_horizon: int, **kwargs) -> Dict[str, torch.Tensor]:
        pseudo_traj = self._pseudo_ego.unroll_trajectory(torch.ones((t_horizon, 2)) * 0.01, dt=self.dt)
        return self._build_connected_graph(t_horizon=t_horizon, ego_trajectory=pseudo_traj, **kwargs)

    @staticmethod
    def trajectory_from_distribution(
        gmm,
        num_output_modes: int,
        dt: float,
        t_horizon: int,
        t_start: float = 0.0,
        return_more: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
        """Transform the Trajectron model GMM distribution to a trajectory.
        The output of the Trajectron model is a Gaussian Mixture Model (GMM) with mean and log_variance (among several
        other properties) for each of the 25 modes. Since `num_modes` < 25 in a first step the most important modes
        will be selected, by using the inverse of the length of the variance vector, i.e.

        .. math:: w = \sum_t^T \omega_t 1 / \sqrt{\sigma_{t,X}^2 + \sigma_{t,Y}^2}

        The importance vector omega is introduced in order to encounter the importance of uncertainty with respect to
        the evolution in time. A high uncertainty at the beginning of the trajectory is worse in terms of planning than
        at its end. The importance vector is a simple linear function going from 1 to 0.2 uniformly over `t_horizon`.

        mus.shape: (num_ados = 1, 1, t_horizon, num_modes, 2)
        log_sigma.shape: (num_ados = 1, 1, t_horizon, num_modes, 2)
        """
        assert t_horizon >= 1
        with torch.no_grad():
            mus = gmm.mus.permute(0, 1, 3, 2, 4)[0, 0, :, :, 0:2]
            log_sigmas = gmm.log_sigmas.permute(0, 1, 3, 2, 4)[0, 0, :, :, 0:2]

            # Determine weight of every mode and get indices of `N = num_modes` highest weights. Since instead of
            # the inverse of the norm of `log_sigmas` the norm is used directly, the lowest results correspond to the
            # most important modes, i.e. highest weights.
            log_sigmas_norm = torch.norm(log_sigmas, dim=2)
            if t_horizon > 1:
                importance = torch.linspace(1.0, 0.2, steps=t_horizon - 1)
                weights_inv = torch.sum(torch.mul(log_sigmas_norm, importance), dim=1).numpy()
            else:
                weights_inv = log_sigmas_norm.view(-1,)

            weights_indices = np.argpartition(weights_inv, kth=num_output_modes)[:num_output_modes]
            weights_np = 1 / weights_inv[weights_indices]
            weights_np = weights_np / np.linalg.norm(weights_np)  # normalization
            weights = torch.from_numpy(weights_np)

        # Write means of highest weight modes in output trajectory. While the means merely describe the positions
        # (2D path points) the (mean) velocity can be determined by computing the finite time difference between
        # subsequent path points, since the ados are assumed to be single integrators.
        trajectory = torch.zeros((num_output_modes, t_horizon, 5))
        trajectory[:, 1:, 0:2] = mus[weights_indices, :, :]
        trajectory[:, 1:-1, 2:4] = (mus[weights_indices, 1:, :] - mus[weights_indices, 0:-1, :]) / dt
        trajectory[:, :, 4] = torch.linspace(t_start, t_start + t_horizon * dt, steps=t_horizon)
        return (trajectory, weights) if not return_more else (trajectory, weights, weights_indices)

    def detach(self):
        """Detaching the whole graph (which is the whole neural network) might be hard. Therefore just rebuilt it
        from scratch completely, using the most up-to-date states of the agents. """
        super(Trajectron, self).detach()

        # Reset internal scene representation.
        self._gt_scene.nodes = []
        self._gt_scene.robot = None

        # Add all agents to the scene again.
        self._add_ego_to_graph(ego=self.ego if self.ego is not None else self._pseudo_ego)
        for i in range(self.num_ados):
            ghosts_ado = self.ghosts_by_ado_index(ado_index=i)
            ado_id, _ = self.split_ghost_id(ghost_id=ghosts_ado[0].id)
            ado_history = ghosts_ado[0].agent.history
            self._add_ado_to_graph(ado_history=ado_history, ado_id=ado_id)

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

        # Pass online environment to internal trajectron model.
        self.trajectron.set_environment(online_env, init_time_step)
        return online_env

    ###########################################################################
    # Utility #################################################################
    ###########################################################################
    def load_and_check_configuration(self, config_path: str) -> Dict[str, Any]:
        self.import_modules()
        from argument_parser import args

        # Load configuration files.
        config = {"trajectron_model_path": build_os_path(f"config/trajectron_models/{env_trajectron_model[0]}"),
                  "trajectron_model_iteration": env_trajectron_model[1]}
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

    def _create_node_data(self, state_history: torch.Tensor, accelerations: torch.Tensor = None) -> pd.DataFrame:
        assert check_ego_trajectory(state_history, pos_and_vel_only=True)

        t_horizon = state_history.shape[0]
        if accelerations is None:
            dd = Derivative2(horizon=t_horizon, dt=self.dt, velocity=True)
            accelerations = dd.compute(state_history[:, 2:4])
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
        module_path = build_os_path("external/GenTrajectron/code", make_dir=False, free=False)
        assert os.path.isdir(module_path)
        return module_path

    @property
    def config(self) -> Dict[str, Any]:
        return self._config
