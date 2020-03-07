import json
import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from mantrap.agents.agent import Agent
from mantrap.agents import IntegratorDTAgent
from mantrap.constants import sim_trajectron_model
from mantrap.simulation.simulation import GraphBasedSimulation
from mantrap.utility.io import build_os_path
from mantrap.utility.maths import Derivative2
from mantrap.utility.shaping import check_ego_trajectory


# TODO: online_trajectron.py:174 ==> no use of `std` as argument ??
class Trajectron(GraphBasedSimulation):

    def __init__(self, ego_type: Agent.__class__, ego_kwargs: Dict[str, Any], **sim_kwargs):
        assert ego_type is not None

        self._config = self.load_and_check_configuration(config_path=build_os_path("config/trajectron.json"))
        super(Trajectron, self).__init__(ego_type, ego_kwargs, dt=self.config["dt"], **sim_kwargs)

        # For prediction un-conditioned on the ego (`predict_wo_ego()`) we need a pseudo-ego trajectory, since the
        # input dimensions for the trajectron have to stay the same.
        self._pseudo_ego = ego_type(position=torch.tensor(self.axes[0]))

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
        from data import Node
        node_data = self._create_node_data(state_history=self.ego.history)
        node = Node(node_type=self._gt_env.NodeType.ROBOT, node_id=self.ego.id, data=node_data, is_robot=True)
        self._gt_scene.robot = node
        self._gt_scene.nodes.append(node)
        self._online_env = self.create_online_env(env=self._gt_env, scene=self._gt_scene)

    ###########################################################################
    # Prediction ##############################################################
    ###########################################################################
    def predict_w_controls(self, controls: torch.Tensor, return_more: bool = False, **graph_kwargs) -> torch.Tensor:
        ego_trajectory = self.ego.unroll_trajectory(controls=controls, dt=self.dt)
        return self.predict_w_trajectory(trajectory=ego_trajectory, return_more=return_more)

    def predict_w_trajectory(self, trajectory: torch.Tensor, return_more: bool = False, **graph_kwargs) -> torch.Tensor:
        assert not return_more  # not implemented in the moment (!)
        assert check_ego_trajectory(trajectory, pos_and_vel_only=True)

        dd = Derivative2(horizon=trajectory.shape[0], dt=self.dt, velocity=True)
        trajectory_w_acc = torch.cat((trajectory[:, 0:4], dd.compute(trajectory[:, 2:4])), dim=1)
        distribution, _ = self.trajectron.incremental_forward(
            new_inputs_dict=self._gt_scene.get_clipped_pos_dict(0, self.config["state"]),
            prediction_horizon=trajectory.shape[0] - 1,
            num_samples=1,
            full_dist=True,
            robot_present_and_future=trajectory_w_acc
        )
        return distribution

    def predict_wo_ego(self, t_horizon: int, return_more: bool = False, **graph_kwargs) -> torch.Tensor:
        """The Trajectron model requires to get some robot position. Therefore, in order to minimize the
        impact of the ego robot on the trajectories (since the prediction should be not conditioned on the robot)
        some pseudo trajectory is used, which is very far distant from the actual scene.

        Within the trajectory optimisation the ado's trajectories conditioned on the robot's planned motion are
        compared with their trajectories without taking any robot into account. So when both the conditioned and
        un-conditioned model for these predictions would be used, and they would be behavioral different, it would
        lead to some base difference (even if there is no robot affecting some ado at all) which might be larger in
        scale than the difference the conditioning on the robot makes. Then minimizing the difference would miss the
        goal of minimizing interaction.
        """
        pseudo_traj = self._pseudo_ego.unroll_trajectory(torch.ones((t_horizon, 2)) * 0.01, dt=self.dt)
        return self.predict_w_trajectory(trajectory=pseudo_traj, return_more=return_more)

    ###########################################################################
    # Scene ###################################################################
    ###########################################################################
    def add_ado(self, **ado_kwargs):
        super(Trajectron, self).add_ado(type=IntegratorDTAgent, **ado_kwargs)
        from data import Node

        node_data = self._create_node_data(state_history=self.ados[-1].history)
        node = Node(node_type=self._gt_env.NodeType.PEDESTRIAN, node_id=self.ados[-1].id, data=node_data)
        self._gt_scene.nodes.append(node)
        self._online_env = self.create_online_env(env=self._gt_env, scene=self._gt_scene)

    ###########################################################################
    # Simulation Graph ########################################################
    ###########################################################################
    def build_connected_graph(self, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

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
        config = {"trajectron_model_path": build_os_path(f"config/trajectron_models/{sim_trajectron_model[0]}"),
                  "trajectron_model_iteration": sim_trajectron_model[1]}
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
