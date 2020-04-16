from abc import ABC, abstractmethod
from copy import deepcopy
import logging
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from mantrap.agents.agent import Agent
from mantrap.constants import *
from mantrap.utility.io import build_os_path
from mantrap.utility.shaping import (
    check_ego_controls,
    check_ego_state,
    check_ego_trajectory,
    check_ado_controls,
    check_ado_states,
    check_ado_trajectories,
    check_weights,
)


class GraphBasedEnvironment(ABC):
    """General environment engine for obstacle-free, interaction-aware, probabilistic and multi-modal agent
    environments. As used in a robotics use-case the environment separates between the ego-agent (the robot) and
    ado-agents (other agents in the scene which are not the robot).

    In order to deal with multi-modality the environment uses so called "ghosts", which are weighted representations
    of an agent. If for example an agent has two modes, two ghosts objects will be assigned to this agent, while being
    treated independently from each other (just not interacting with each other).

    To store only the ghosts themselves and not the agents was avoids storage overhead and makes it easier to simulate
    several independent modes at the same time. However if only ado representations are required, the most important
    mode of each ado can be collected using the `ados_most_important_mode()` method.

    The internal states basically are the states of the ego and ados and can only be changed by either using the
    `step()` or `step_reset()` function, which simulate how the environment reacts based on some action performed by
    the ego or resets it directly to some given states.

    The simulated world is two-dimensional and defined in the area limited by the passed `x_axis` and `y_axis`. It has
    a constant environment time-step `dt`.

    :param ego_type: agent class of ego agent (should be agent child-class).
    :param ego_kwargs: initialization arguments of ego agent such as position, velocity, etc.
    :param x_axis: environment environment limitation in x-direction.
    :param y_axis: environment environment limitation in y-direction.
    :param dt: environment time-step [s].
    :param verbose: debugging flag (-1: nothing, 0: logging, 1: +printing, 2: +plot scenes, 3: +plot optimization).
    :param config_name: configuration name of initialized environment (for logging purposes only).
    """

    class Ghost:
        """The Ghost class is a container for an object representing one mode of some ado agent.

        A ghost is defined by its underlying agent (the actual ado), the mode's weight (importance) and its identifier,
        which is constructed as defined in the `GraphBasedEnvironment::build_ghost_id` method. While it is widely
        not constant, not permitting the underlying agent and identifier to change, its weight might change over
        prediction time, e.g. due to a change in the environment scene.

        Treating the modes as independent objects grants the chance to quickly iterate over all modes, easily include
        the interaction between different permutations of modes and detaching from computation graphs.
        """
        def __init__(self, agent: Agent, weight: float, identifier: str, **params):
            self._agent = agent
            self._weight = weight
            self._identifier = identifier
            self._params = params

        @property
        def agent(self) -> Agent:
            return self._agent

        @property
        def weight(self) -> float:
            return self._weight

        @weight.setter
        def weight(self, weight_new: float):
            assert 0 <= weight_new
            self._weight = weight_new

        @property
        def id(self) -> str:
            return self._identifier

        @property
        def params(self) -> Dict[str, float]:
            return self._params

    ###########################################################################
    # Initialization ##########################################################
    ###########################################################################
    def __init__(
        self,
        ego_type: Agent.__class__ = None,
        ego_kwargs: Dict[str, Any] = None,
        x_axis: Tuple[float, float] = ENV_X_AXIS_DEFAULT,
        y_axis: Tuple[float, float] = ENV_Y_AXIS_DEFAULT,
        dt: float = ENV_DT_DEFAULT,
        verbose: int = -1,
        config_name: str = CONFIG_UNKNOWN
    ):
        assert x_axis[0] < x_axis[1], "x axis must be in form (x_min, x_max)"
        assert y_axis[0] < y_axis[1], "y axis must be in form (y_min, y_max)"
        assert dt > 0.0, "time-step must be larger than 0"

        self._ego = ego_type(**ego_kwargs, identifier=ID_EGO) if ego_type is not None else None
        self._ado_ghosts = []
        self._num_ado_modes = 0
        self._ado_ids = []
        self._ado_ghost_ids = []  # quick access only

        # Dictionary of environment parameters.
        self._env_params = dict()
        self._env_params[PARAMS_X_AXIS] = x_axis
        self._env_params[PARAMS_Y_AXIS] = y_axis
        self._env_params[PARAMS_CONFIG] = config_name
        self._env_params[PARAMS_VERBOSE] = verbose
        self._dt = dt
        self._time = 0

        # Perform sanity check for environment and agents.
        assert self.sanity_check()

    ###########################################################################
    # Simulation step #########################################################
    ###########################################################################
    def step(self, ego_control: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Run environment step (time-step = dt). Update state and history of ados and ego. Also reset environment time
        to time_new = time + dt. The difference to predict() is two-fold: Firstly, step() is only going forward
        one time-step at a time, not in general `t_horizon` steps, secondly, step() changes the actual agent states
        in the environment while predict() copies all agents and changes the states of these copies (so the actual
        agent states remain unchanged).

        :param ego_control: planned ego control input for current time step (1, 2).
        :returns: ado_states (num_ados, num_modes, 1, 5), ego_next_state (5) in next time step.
        """
        assert check_ego_controls(ego_control, t_horizon=1)
        self._time = self._time + self.dt

        # Unroll future ego trajectory, which is surely deterministic and certain due to the deterministic dynamics
        # assumption. Update ego based on the first action of the input ego policy.
        self._ego.update(ego_control.flatten(), dt=self.dt)
        logging.info(f"env {self.name} step @t={self.time} [ego]: action={ego_control.tolist()}")
        logging.info(f"env {self.name} step @t={self.time} [ego_{self._ego.id}]: state={self.ego.state.tolist()}")

        # Predict the next step in the environment by forward environment.
        _, ado_controls, weights = self.predict_w_controls(ego_controls=ego_control, return_more=True)

        # Update ados by forward simulate them and determining their most likely policies. Therefore predict the
        # ado states at the next time step as well as the probabilities (weights) of them occurring. Then sample one
        # mode (given these weights) and update the ados as that sampled mode.
        # The base state should be the same between all modes, therefore update all mode states according to the
        # one sampled mode policy.
        weights = weights.detach().numpy()
        ado_states = torch.zeros((self.num_ados, 5))  # deterministic update (!)
        sampled_modes = {}
        for ado_id in self.ado_ids:
            i_ado = self.index_ado_id(ado_id=ado_id)
            assert weights[i_ado, :].size == self.num_modes
            weights_normed = weights[i_ado, :] / weights[i_ado, :].sum()  # normalize probability
            choices = np.arange(start=0, stop=self.num_modes)
            sampled_modes[ado_id] = np.random.choice(choices, p=weights_normed)

        # Now update the internal ghost representations accordingly, every ghost originating from the ado should now
        # be "synchronized", i.e. have the same current state.
        for j in range(self.num_ghosts):
            ado_id, _ = self.split_ghost_id(ghost_id=self.ghosts[j].id)
            i_ado = self.index_ado_id(ado_id=ado_id)
            self._ado_ghosts[j].agent.update(action=ado_controls[i_ado, sampled_modes[ado_id], 0, :], dt=self.dt)
            ado_states[i_ado, :] = self.ghosts[j].agent.state_with_time  # TODO: repetitive !
            logging.info(f"env {self.name} step @t={self.time} [ado_{ado_id}]: state={ado_states[i_ado].tolist()}")

        # Detach agents from graph in order to keep independence between subsequent runs. Afterwards perform sanity
        # check for environment and agents.
        self.detach()
        assert self.sanity_check()

        assert check_ado_states(x=ado_states, num_ados=self.num_ados, enforce_temporal=True)
        return ado_states.detach(), self.ego.state_with_time.detach()  # otherwise no scene independence (!)

    def step_reset(self, ego_state_next: Union[torch.Tensor, None], ado_states_next: Union[torch.Tensor, None]):
        """Run environment step (time-step = dt). Instead of predicting the behaviour of every agent in the scene, it
        is given as an input and the agents are merely updated. All the ghosts (modes of an ado) will collapse to the
        same given state, since the update is deterministic.

        :param ego_state_next: ego state for next time step (5).
        :param ado_states_next: ado states for next time step (num_ados, num_modes, 1, 5).
        """
        self._time = self._time + self.dt

        # Reset ego agent (if there is an ego in the scene), otherwise just do not reset it.
        if ego_state_next is not None:
            assert check_ego_state(ego_state_next, enforce_temporal=True)
            self._ego.reset(state=ego_state_next, history=None)  # new state is appended

        # Reset ado agents, each mode similarly, if `ado_states_next` is None just do not reset them. When resetting
        # with `history=None` the new state is appended automatically.
        if ado_states_next is not None:
            assert check_ado_states(ado_states_next, num_ados=self.num_ados, enforce_temporal=True)
            for m_ghost in range(self.num_ghosts):
                m_ado, _ = self.convert_ghost_id(ghost_id=self.ghosts[m_ghost].id)
                self._ado_ghosts[m_ghost].agent.reset(ado_states_next[m_ado, :], history=None)

        # Detach agents from graph in order to keep independence between subsequent runs. Afterwards perform sanity
        # check for environment and agents.
        self.detach()
        assert self.sanity_check()

    ###########################################################################
    # Prediction ##############################################################
    ###########################################################################
    def predict_w_controls(self, ego_controls: torch.Tensor, return_more: bool = False, **kwargs) -> torch.Tensor:
        """Predict the environments future for the given time horizon (discrete time).
        The internal prediction model is dependent on the exact implementation of the internal interaction model
        between the ados with each other and between the ados and the ego. The implementation therefore is specific
        to each child-class.

        :param ego_controls: ego control input (pred_horizon, 2).
        :param return_more: return the system inputs (at every time -> trajectory) and probabilities of each mode.
        :param kwargs: additional arguments for graph construction.
        :return: predicted trajectories for ados in the scene (either one or multiple for each ado).
        """
        assert self.ego is not None
        assert check_ego_controls(ego_controls)
        assert self.sanity_check()

        ego_trajectory = self.ego.unroll_trajectory(controls=ego_controls, dt=self.dt)
        graph = self.build_connected_graph(ego_trajectory, ego_grad=False, **kwargs)
        return self.transcribe_graph(graph, t_horizon=ego_controls.shape[0] + 1, return_more=return_more)

    def predict_w_trajectory(self, ego_trajectory: torch.Tensor, return_more: bool = False, **kwargs) -> torch.Tensor:
        """Predict the environments future for the given time horizon (discrete time).
        The internal prediction model is dependent on the exact implementation of the internal interaction model
        between the ados with each other and between the ados and the ego. The implementation therefore is specific
        to each child-class.

        :param ego_trajectory: ego trajectory (pred_horizon, 4).
        :param return_more: return the system inputs (at every time -> trajectory) and probabilities of each mode.
        :param kwargs: additional arguments for graph construction.
        :return: predicted trajectories for ados in the scene (either one or multiple for each ado).
        """
        assert self.ego is not None
        assert check_ego_trajectory(x=ego_trajectory, pos_and_vel_only=True)
        assert self.sanity_check()

        graph = self.build_connected_graph(ego_trajectory=ego_trajectory, ego_grad=False, **kwargs)
        return self.transcribe_graph(graph, t_horizon=ego_trajectory.shape[0], return_more=return_more)

    def predict_wo_ego(self, t_horizon: int, return_more: bool = False, **kwargs) -> torch.Tensor:
        """Predict the environments future for the given time horizon (discrete time).
        The internal prediction model is dependent on the exact implementation of the internal interaction model
        between the ados while ignoring the ego.

        :param t_horizon: prediction horizon, number of discrete time-steps.
        :param return_more: return the system inputs (at every time -> trajectory) and probabilities of each mode.
        :param kwargs: additional arguments for graph construction.
        :return: predicted trajectories for ados in the scene (either one or multiple for each ado).
        """
        assert t_horizon > 0
        assert self.sanity_check()

        graph = self.build_connected_graph_wo_ego(t_horizon=t_horizon, **kwargs)
        return self.transcribe_graph(graph, t_horizon=t_horizon, return_more=return_more)

    ###########################################################################
    # Scene ###################################################################
    ###########################################################################
    def states(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return current states of ego and ado agents in the scene. Since the current state is known for every
        ado the states are deterministic and uni-modal. States are returned as vector including temporal dimension.

        :returns: ego state vector including temporal dimension (5).
        :returns: ado state vectors including temporal dimension (num_ados, 5).
        """
        ado_states = torch.zeros((self.num_ados, 5))
        for ado_id in self.ado_ids:
            m_ado = self.index_ado_id(ado_id=ado_id)
            m_ghost = self.convert_ado_id(ado_id=ado_id, mode_index=0)  # 0 independent from num_modes
            ado_states[m_ado, :] = self.ghosts[m_ghost].agent.state_with_time
        ego_state = self.ego.state_with_time if self.ego is not None else None

        if ego_state is not None:
            assert check_ego_state(x=ego_state, enforce_temporal=True)
        assert check_ado_states(x=ado_states, enforce_temporal=True, num_ados=self.num_ados)
        return ego_state, ado_states

    def add_ado(
        self,
        ado_type: Agent.__class__ = None,
        num_modes: int = 1,
        weights: List[float] = None,
        arg_list: List[Dict] = None,
        **ado_kwargs
    ) -> Agent:
        """Add (multi-modal) ado (i.e. non-robot) agent to environment.
        While the ego is added to the environment during initialization, the ado agents have to be added afterwards,
        individually. To do so for each mode an agent is initialized using the passed initialization arguments and
        appended to the internal list of ghosts, while staying assignable to the original ado agent by id, i.e.
        ghost_id = ado_id + mode_index.
        Thereby the ghosts are sorted with decreasing level of importance, i.e. decreasing weight, so that the first
        ghost in the list of added ghosts for this agent always is the most important one.

        :param ado_type: agent class of creating ado (has to be subclass of Agent-class in agents/).
        :param num_modes: number of modes of multi-modal ado agent (>=1).
        :param weights: mode weight vector, default = uniform distribution.
        :param arg_list: initialization arguments for each mode.
        """
        assert ado_type is not None and type(ado_type) == Agent.__class__
        ado = ado_type(**ado_kwargs)
        self._ado_ids.append(ado.id)

        # Append ado to internal list of ados and rebuilt the graph (could be also extended but small computational
        # to actually rebuild it).
        assert self.axes[0][0] <= ado.position[0] <= self.axes[0][1]
        assert self.axes[1][0] <= ado.position[1] <= self.axes[1][1]
        if self._num_ado_modes == 0:
            self._num_ado_modes = num_modes
        assert num_modes == self.num_modes  # all ados should have same number of modes
        if not self.is_multi_modal:
            assert num_modes == 1  # environment does not support multi-modality

        # Append the created ado for every mode. When no weights are given, then initialize with a uniform weight
        # distribution between the modes. If not sort the modes by order of decreasing weight. For uniform
        # distributions no modes are not sorted due to the order - inversion during the sorting process, which
        # is problematic e.g. when copying the environment.
        arg_list = arg_list if arg_list is not None else [dict()] * num_modes
        weights = weights if weights is not None else (torch.ones(num_modes) / num_modes).tolist()
        if not all([abs(weights[0] - weights[i]) < 1e-6 for i in range(num_modes)]):  # = non-uniform
            index_sorted = list(reversed(np.argsort(weights)))  # so that decreasing order
            arg_list = [arg_list[k] for k in index_sorted]
            weights = [weights[k] for k in index_sorted]
        assert len(arg_list) == len(weights) == num_modes

        for i in range(num_modes):
            ado = deepcopy(ado)
            gid = self.build_ghost_id(ado_id=ado.id, mode_index=i)
            self._ado_ghosts.append(self.Ghost(ado, weight=weights[i], identifier=gid, **arg_list[i]))
            self._ado_ghost_ids.append(gid)

        # Perform sanity check for environment and agents.
        assert self.sanity_check()
        return ado

    def ados_most_important_mode(self) -> List[Ghost]:
        """Return a list of the most important ghosts, i.e. the ones with the highest weight, for each ado, by
        exploiting the way they are added to the list of ghosts (decreasing weights so that the first ghost
        is the most important one).
        The functionality of the method can be checked by comparing the ado ids of the selected most important ghosts.
        Since the ids are unique the only way to get N different ado ids in the list is by having selected ghosts
        from N different ados.
        """
        ghost_max = [self.ghosts[i * self.num_modes + 0] for i in range(self.num_ados)]

        assert len(np.unique([ghost.id for ghost in ghost_max])) == self.num_ados
        return ghost_max

    def ghosts_by_ado_index(self, ado_index: int) -> List[Ghost]:
        assert 0 <= ado_index < self.num_ados
        return self._ado_ghosts[ado_index * self.num_modes:(ado_index + 1)*self.num_modes]

    def ghosts_by_ado_id(self, ado_id: str) -> List[Ghost]:
        index = self.index_ado_id(ado_id=ado_id)
        return self.ghosts_by_ado_index(ado_index=index)

    ###########################################################################
    # Ghost ID ################################################################
    ###########################################################################
    @staticmethod
    def build_ghost_id(ado_id: str, mode_index: int) -> str:
        return ado_id + "_" + str(mode_index)

    @staticmethod
    def split_ghost_id(ghost_id: str) -> Tuple[str, int]:
        ado_id, mode_index = ghost_id.split("_")
        return ado_id, int(mode_index)

    def index_ado_id(self, ado_id: str) -> int:
        return self.ado_ids.index(ado_id)

    def index_ghost_id(self, ghost_id: str) -> int:
        return self.ghost_ids.index(ghost_id)

    def convert_ado_id(self, ado_id: str, mode_index: int) -> int:
        ado_index = self.index_ado_id(ado_id=ado_id)
        return ado_index * self.num_modes + mode_index

    def convert_ghost_id(self, ghost_id: str) -> Tuple[int, int]:
        ado_id, mode_index = self.split_ghost_id(ghost_id)
        return self.ado_ids.index(ado_id), mode_index

    ###########################################################################
    # Simulation graph ########################################################
    ###########################################################################
    def build_connected_graph(self, ego_trajectory: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Build differentiable graph for predictions over multiple time-steps. For the sake of differentiability
        the computation for the nth time-step cannot be done iteratively, i.e. by determining the current states and
        using the resulting values for computing the next time-step's results in a Markovian manner. Instead the whole
        graph (which is the whole computation) has to be built over n time-steps and evaluated at once by forward pass.

        For building the graph the graphs for each single time-step is built independently while being connected
        using the outputs of the previous time-step and an input for the current time-step. This is quite heavy in
        terms of computational effort and space, however end-to-end-differentiable.

        Build the graph conditioned on some `ego_trajectory`, which is assumed to be fix while the ados in the scene
        behave accordingly, i.e. in reaction to the ego's trajectory. The resulting graph will then contain states
        and controls for every agent in the scene for t in [0, t_horizon], which t_horizon = length of ego trajectory.

        :param ego_trajectory: ego's trajectory (t_horizon, 5).
        :kwargs: additional graph building arguments.
        :return: dictionary over every state of every agent in the scene for t in [0, t_horizon].
        """
        assert check_ego_trajectory(ego_trajectory, pos_and_vel_only=True)
        assert self.ego is not None
        graph = self._build_connected_graph(ego_trajectory=ego_trajectory, **kwargs)
        assert self.check_graph(graph, t_horizon=ego_trajectory.shape[0], include_ego=True)
        return graph

    @abstractmethod
    def _build_connected_graph(self, ego_trajectory: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Build a connected graph based on the ego's trajectory.

        The graph should span over the time-horizon of the length of the ego's trajectory and contain the state
        (position, velocity) and "controls" of every ghost in the scene as well as the ego's states itself. When
        possible the graph should be differentiable, such that finding some gradient between the outputted ado
        states and the inputted ego trajectory is determinable.

        :param ego_trajectory: ego's trajectory (t_horizon, 5).
        :return: dictionary over every state of every agent in the scene for t in [0, t_horizon].
        """
        raise NotImplementedError

    def build_connected_graph_wo_ego(self, t_horizon: int, **kwargs) -> Dict[str, torch.Tensor]:
        """Build differentiable graph for predictions over multiple time-steps. For the sake of differentiability
        the computation for the nth time-step cannot be done iteratively, i.e. by determining the current states and
        using the resulting values for computing the next time-step's results in a Markovian manner. Instead the whole
        graph (which is the whole computation) has to be built over n time-steps and evaluated at once by forward pass.

        For building the graph the graphs for each single time-step is built independently while being connected
        using the outputs of the previous time-step and an input for the current time-step. This is quite heavy in
        terms of computational effort and space, however end-to-end-differentiable.

        Build the graph as if no ego robot would be in the scene, whether or not an ego agent is defined internally.
        Therefore, merely the time-horizon for the predictions (= number of prediction time-steps) is passed.

        :param t_horizon: number of prediction time-steps.
        :kwargs: additional graph building arguments.
        :return: dictionary over every state and control of every ado in the scene for t in [0, t_horizon].
        """
        assert t_horizon > 0
        graph = self._build_connected_graph_wo_ego(t_horizon=t_horizon, **kwargs)
        assert self.check_graph(graph, t_horizon=t_horizon, include_ego=False)
        return graph

    @abstractmethod
    def _build_connected_graph_wo_ego(self, t_horizon: int, **kwargs) -> Dict[str, torch.Tensor]:
        """Build a connected graph over `t_horizon` time-steps for ados only.

        The graph should span over the time-horizon of the inputted number of time-steps and contain the state
        (position, velocity) and "controls" of every ghost in the scene as well as the ego's states itself. When
        possible the graph should be differentiable, such that finding some gradient between the outputted ado
        states and the inputted ego trajectory is determinable.

        :param t_horizon: number of prediction time-steps.
        :return: dictionary over every state of every ado in the scene for t in [0, t_horizon].
        """
        raise NotImplementedError

    def check_graph(self, graph: Dict[str, torch.Tensor], t_horizon: int, include_ego: bool = True) -> bool:
        """Check connected graph keys for completeness. The graph is connected for several (discrete) time-steps,
        from 0 to `t_horizon` and should contain a state and control for every agent in the scene for these
        points in time. As the graph is assumed to be complete in keys, a non-complete graph cannot be used for
        further computation.
        """
        for ghost_id in self.ghost_ids:
            assert all([f"{ghost_id}_{k}_{GK_POSITION}" in graph.keys() for k in range(t_horizon)])
            assert all([f"{ghost_id}_{k}_{GK_VELOCITY}" in graph.keys() for k in range(t_horizon)])
            assert all([f"{ghost_id}_{k}_{GK_CONTROL}" in graph.keys() for k in range(t_horizon)])

        if include_ego:
            assert all([f"{ID_EGO}_{k}_{GK_POSITION}" in graph.keys() for k in range(t_horizon)])
            assert all([f"{ID_EGO}_{k}_{GK_VELOCITY}" in graph.keys() for k in range(t_horizon)])

        return True

    def write_state_to_graph(
        self, ego_state: torch.Tensor = None, k: int = 0, ado_grad: bool = False, ego_grad: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Given some state for the ego (and the internal state of every ghost in the scene), initialize a graph
        and write these states into it. A graph is a dictionary of tensors for this current state.

        :param ego_state: current state of ego robot (might deviate from internal state) (5).
        :param k: time-step count for graph key description.
        :param ado_grad: flag whether the ado-related tensors should originate a gradient-chain.
        :param ego_grad: flag whether the ego-related tensors should originate a gradient-chain.
        """
        graph = {}

        if ego_state is not None:
            assert check_ego_state(x=ego_state, enforce_temporal=False)
            graph[f"{ID_EGO}_{k}_{GK_POSITION}"] = ego_state[0:2]
            graph[f"{ID_EGO}_{k}_{GK_VELOCITY}"] = ego_state[2:4]

            if ego_grad and not graph[f"{ID_EGO}_{k}_{GK_POSITION}"].requires_grad:  # if require_grad is set
                graph[f"{ID_EGO}_{k}_{GK_POSITION}"].requires_grad = True
                graph[f"{ID_EGO}_{k}_{GK_VELOCITY}"].requires_grad = True

        for ghost in self.ghosts:
            graph[f"{ghost.id}_{k}_{GK_POSITION}"] = ghost.agent.position
            graph[f"{ghost.id}_{k}_{GK_VELOCITY}"] = ghost.agent.velocity
            if ado_grad and graph[f"{ghost.id}_{k}_{GK_POSITION}"].requires_grad is not True:
                graph[f"{ghost.id}_{k}_{GK_POSITION}"].requires_grad = True
                graph[f"{ghost.id}_{k}_{GK_VELOCITY}"].requires_grad = True

        return graph

    def transcribe_graph(
        self,
        graph: Dict[str, torch.Tensor],
        t_horizon: int,
        return_more: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Transcribe states stored in a graph into trajectories and controls.

        A connected graph contains the states and controls of every agent in the scene (ego & ados) for every
        t in [0, t_horizon]. Read these states and controls to build a trajectory for the ados in the scene.

        :param graph: connected input graph.
        :param t_horizon: time-horizon to build trajectory (number of discrete time-steps).
        :param return_more: return ado-trajectory, -controls and -weights or trajectory only.
        """
        ado_controls = torch.zeros((self.num_ados, self.num_modes, t_horizon - 1, 2))
        ado_trajectories = torch.zeros((self.num_ados, self.num_modes, t_horizon, 5))
        weights = torch.zeros((self.num_ados, self.num_modes))

        for ghost in self.ghosts:
            m_ado, m_mode = self.convert_ghost_id(ghost_id=ghost.id)
            for t in range(t_horizon):
                ado_trajectories[m_ado, m_mode, t, 0:2] = graph[f"{ghost.id}_{t}_{GK_POSITION}"]
                ado_trajectories[m_ado, m_mode, t, 2:4] = graph[f"{ghost.id}_{t}_{GK_VELOCITY}"]
                ado_trajectories[m_ado, m_mode, t, -1] = self.time + self.dt * t
                if t < t_horizon - 1:
                    ado_controls[m_ado, m_mode, t, :] = graph[f"{ghost.id}_{t}_{GK_CONTROL}"]
            weights[m_ado, m_mode] = ghost.weight

        # Check output shapes. Besides, since all modes originate in the same ado, their first state (t = t0) should
        # be equivalent, namely the current environment's state, since it is deterministic.
        num_modes, num_ados = self.num_modes, self.num_ados
        assert check_ado_controls(ado_controls, num_ados=num_ados, num_modes=num_modes, t_horizon=t_horizon-1)
        assert check_weights(weights, num_ados=num_ados, num_modes=num_modes)
        assert check_ado_trajectories(ado_trajectories, ados=num_ados, t_horizon=t_horizon, modes=num_modes)
        for ghost in self.ghosts:
            m_ado, m_mode = self.convert_ghost_id(ghost_id=ghost.id)
            assert torch.all(torch.isclose(ado_trajectories[m_ado, m_mode, 0, :], ado_trajectories[m_ado, 0, 0, :]))
        return ado_trajectories if not return_more else (ado_trajectories, ado_controls, weights)

    def detach(self):
        """Detach all internal agents (ego and all ado ghosts) from computation graph. This is sometimes required to
        completely separate subsequent computations in PyTorch."""
        self._ego.detach()
        for m in range(self.num_ghosts):
            self.ghosts[m].agent.detach()

    ###########################################################################
    # Operators ###############################################################
    ###########################################################################
    def copy(self, env_type: 'GraphBasedEnvironment'.__class__ = None) -> 'GraphBasedEnvironment':
        """Create copy of environment.

        However just using deepcopy is not supported for tensors that are not detached
        from the PyTorch computation graph. Therefore re-initialize the objects such as the agents in the
        environment and reset their state to the internal current state.

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
                ego_kwargs = {"position": position, "velocity": velocity, "history": history, "time": self.time}

            env_copy = env_type(ego_type, ego_kwargs, dt=self.dt, **self._env_params)

            # Add internal ado agents to newly created environment.
            env_copy = self._copy_ados(env_copy=env_copy)

        assert self.same_initial_conditions(other=env_copy)
        assert env_copy.sanity_check()
        return env_copy

    def _copy_ados(self, env_copy: 'GraphBasedEnvironment') -> 'GraphBasedEnvironment':
        for i in range(self.num_ados):
            ghosts_ado = self.ghosts_by_ado_index(ado_index=i)
            ado_id, _ = self.split_ghost_id(ghost_id=ghosts_ado[0].id)
            env_copy.add_ado(
                position=ghosts_ado[0].agent.position,  # same over all ghosts of same ado
                velocity=ghosts_ado[0].agent.velocity,  # same over all ghosts of same ado
                history=ghosts_ado[0].agent.history,  # same over all ghosts of same ado
                time=self.time,
                weights=[ghost.weight for ghost in ghosts_ado],
                num_modes=self.num_modes,
                identifier=self.split_ghost_id(ghost_id=ghosts_ado[0].id)[0],
            )
        return env_copy

    def same_initial_conditions(self, other: 'GraphBasedEnvironment'):
        """Similar to __eq__() function, but not enforcing parameters of environment to be completely equivalent,
        merely enforcing the initial conditions to be equal, such as states of agents in scene. Hence, all prediction
        depending parameters, namely the number of modes or agent's parameters dont have to be equal.
        """
        assert self.dt == other.dt
        assert self.num_ados == other.num_ados
        assert self.ego == other.ego
        ghosts_equal = [True] * self.num_ados
        ghosts_imp = self.ados_most_important_mode()
        other_imp = other.ados_most_important_mode()
        for i in range(self.num_ados):
            ghosts_equal[i] = ghosts_equal[i] and ghosts_imp[i].agent == other_imp[i].agent
        assert all(ghosts_equal)
        return True

    def sanity_check(self) -> bool:
        """Check for the sanity of the scene and agents.
        In order to check the sanity of the environment some general properties must hold, such as the number and order
        of ghosts in the scene, which should be equivalent to the number of ados times the number of modes (since
        the same number of modes for every ado is being enforced). Also all agents in the scene are checked for their
        sanity.
        """
        assert self.num_ghosts == self.num_ados * self.num_modes
        assert self.num_ados == len(self.ado_ids)
        assert self.num_ghosts == len(self.ghost_ids)

        # Check sanity of all agents in the scene.
        if self.ego is not None:
            self.ego.sanity_check()
        for ghost in self.ghosts:
            ghost.agent.sanity_check()

        # Check for the right order of ghosts, i.e. an order matching between ados and ghosts. Firstly, all ghosts
        # which origin from the same ado must be subsequent and secondly, have the same ado id, which is the one
        # at the kth place of the `ado_ids` array.
        for m_ado, ado_id in enumerate(self.ado_ids):
            weights_per_mode = np.zeros(self.num_modes)
            for m_ghost, ghost in enumerate(self.ghosts[m_ado * self.num_modes:(m_ado + 1) * self.num_modes]):
                ado_id_ghost, _ = self.split_ghost_id(ghost_id=ghost.id)
                assert ado_id == ado_id_ghost
                weights_per_mode[m_ghost] = ghost.weight
            # Check whether ghost order is correct, from highest to smallest weight.
            assert all([weights_per_mode[k] <= weights_per_mode[k - 1] for k in range(1, self.num_modes)])
        return True

    ###########################################################################
    # Visualization ###########################################################
    ###########################################################################
    def visualize_prediction(self, ego_trajectory: torch.Tensor, enforce: bool = False, interactive: bool = False):
        """Visualize the predictions for the scene based on the given ego trajectory.

        In order to be use the general `visualize()` function defined in the `mantrap.evaluation` - package the ego
        and ado trajectories require to be in (num_steps, t_horizon, 5) shape, a representation that allows to
        visualize planned trajectories at multiple points in time (re-planning). However for the purpose of
        plotting the predicted trajectories, there are no changes in planned trajectories. That's why the predicted
        trajectory is repeated to the whole time horizon.
        """
        if self.verbose > 1 or enforce:
            from mantrap.evaluation.visualization import visualize
            assert check_ego_trajectory(x=ego_trajectory)
            t_horizon = ego_trajectory.shape[0]

            # Predict the ado behaviour conditioned on the given ego trajectory.
            ado_trajectories = self.predict_w_trajectory(ego_trajectory=ego_trajectory)
            ado_trajectories_wo = self.predict_wo_ego(t_horizon=t_horizon)

            # Stretch the ego and ado trajectories as described above.
            ego_stretched = torch.zeros((t_horizon, t_horizon, 5))
            ado_stretched = torch.zeros((t_horizon, self.num_ados, self.num_modes, t_horizon, 5))
            ado_stretched_wo = torch.zeros((t_horizon, self.num_ados, self.num_modes, t_horizon, 5))
            for t in range(t_horizon):
                ego_stretched[t, :(t_horizon - t), :] = ego_trajectory[t:t_horizon, :]
                ego_stretched[t, (t_horizon - t):, :] = ego_trajectory[-1, :]
                ado_stretched[t, :, :, :(t_horizon - t), :] = ado_trajectories[:, :, t:t_horizon, :]
                ado_stretched[t, :, :, (t_horizon - t):, :] = ado_trajectories[:, :, -1, :].unsqueeze(dim=2)
                ado_stretched_wo[t, :, :, :(t_horizon - t), :] = ado_trajectories_wo[:, :, t:t_horizon, :]
                ado_stretched_wo[t, :, :, (t_horizon - t):, :] = ado_trajectories_wo[:, :, -1, :].unsqueeze(dim=2)

            return visualize(
                ego_planned=ego_stretched,
                ado_planned=ado_stretched,
                ado_planned_wo=ado_stretched_wo,
                plot_path_only=True,
                env=self,
                file_path=self._visualize_output_format(interactive=interactive, name="prediction")
            )

    def visualize_prediction_wo_ego(self, t_horizon: int, enforce: bool = False, interactive: bool = False):
        """Visualize the predictions for the scene based on the given ego trajectory.

        In order to be use the general `visualize()` function defined in the `mantrap.evaluation` - package the ego
        and ado trajectories require to be in (num_steps, t_horizon, 5) shape, a representation that allows to
        visualize planned trajectories at multiple points in time (re-planning). However for the purpose of
        plotting the predicted trajectories, there are no changes in planned trajectories. That's why the predicted
        trajectory is repeated to the whole time horizon.
        """
        if self.verbose > 1 or enforce:
            from mantrap.evaluation.visualization import visualize

            # Predict the ado behaviour conditioned on the given ego trajectory.
            ado_trajectories_wo = self.predict_wo_ego(t_horizon=t_horizon)

            # Stretch the ego and ado trajectories as described above.
            ado_stretched_wo = torch.zeros((t_horizon, self.num_ados, self.num_modes, t_horizon, 5))
            for t in range(t_horizon):
                ado_stretched_wo[t, :, :, :(t_horizon - t), :] = ado_trajectories_wo[:, :, t:t_horizon, :]
                ado_stretched_wo[t, :, :, (t_horizon - t):, :] = ado_trajectories_wo[:, :, -1, :].unsqueeze(dim=2)

            output_path = self._visualize_output_format(interactive=interactive, name="prediction_wo_ego")
            return visualize(ado_planned_wo=ado_stretched_wo, plot_path_only=True, env=self, file_path=output_path)

    def _visualize_output_format(self, interactive: bool, name: str) -> Union[str, None]:
        """The `visualize()` function enables interactive mode, i.e. returning the video as html5-video directly,
        # instead of saving it as ".gif"-file. Therefore depending on the input flags, set the output path
        # to None (interactive mode) or to an actual path (storing mode). """
        if not interactive:
            output_path = build_os_path(VISUALIZATION_DIRECTORY, make_dir=True, free=False)
            output_path = os.path.join(output_path, f"{self.name}_{name}")
        else:
            output_path = None
        return output_path

    ###########################################################################
    # Ado properties ##########################################################
    ###########################################################################
    @property
    def ado_colors(self) -> List[np.ndarray]:
        ados = self.ados_most_important_mode()
        return [ado.agent.color for ado in ados]

    @property
    def ado_ids(self) -> List[str]:
        return self._ado_ids

    @property
    def num_ados(self) -> int:
        return len(self.ado_ids)

    ###########################################################################
    # Ghost properties ########################################################
    # Per default the ghosts (i.e. the multimodal representations of the ados #
    # are the ados themselves, as the default case is uni-modal. ##############
    ###########################################################################
    @property
    def ghosts(self) -> List[Ghost]:
        return self._ado_ghosts

    @property
    def ghost_ids(self) -> List[str]:
        return self._ado_ghost_ids

    @property
    def num_ghosts(self) -> int:
        return len(self._ado_ghosts)

    @property
    def num_modes(self) -> int:
        return self._num_ado_modes

    ###########################################################################
    # Ego properties ##########################################################
    ###########################################################################
    @property
    def ego(self) -> Union[Agent, None]:
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
    def axes(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return self._env_params[PARAMS_X_AXIS], self._env_params[PARAMS_Y_AXIS]

    @property
    def verbose(self) -> int:
        return self._env_params[PARAMS_VERBOSE]

    @property
    def config_name(self) -> str:
        return self._env_params[PARAMS_CONFIG]

    @property
    def name(self) -> str:
        return self.environment_name + "_" + self.config_name

    ###########################################################################
    # Simulation properties ###################################################
    ###########################################################################
    @property
    def environment_name(self) -> str:
        return self.__class__.__name__.lower()

    @property
    def is_multi_modal(self) -> bool:
        return True

    @property
    def is_deterministic(self) -> bool:
        return True
