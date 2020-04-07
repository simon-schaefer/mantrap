from abc import abstractmethod
from collections import namedtuple
from copy import deepcopy
import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from mantrap.agents.agent import Agent
from mantrap.constants import sim_x_axis_default, sim_y_axis_default, sim_dt_default
from mantrap.utility.io import dict_value_or_default
from mantrap.utility.shaping import check_state, check_trajectories, check_controls, check_weights, check_ego_trajectory, check_ego_controls


class GraphBasedSimulation:
    """General simulation engine for obstacle-free, interaction-aware, probabilistic and multi-modal agent environments.
    As used in a robotics use-case the simulation separates between the ego-agent (the robot) and ado-agents (other
    agents in the scene which are not the robot).

    In order to deal with multi-modality the simulation uses so called "ghosts", which are weighted representations
    of an agent. If for example an agent has two modes, two ghosts objects will be assigned to this agent, while being
    treated independently from each other (just not interacting with each other).

    To store only the ghosts themselves and not the agents was avoids storage overhead and makes it easier to simulate
    several independent modes at the same time. However if only ado representations are required, the most important
    mode of each ado can be collected using the `ados_most_important_mode()` method.

    The internal states basically are the states of the ego and ados and can only be changed by either using the
    `step()` or `step_reset()` function, which simulate how the environment reacts based on some action performed by
    the ego or resets it directly to some given states.

    The simulated world is two-dimensional and defined in the area limited by the passed `x_axis` and `y_axis`. It has
    a constant simulation time-step `dt`.

    :param ego_type: agent class of ego agent (should be agent child-class).
    :param ego_kwargs: initialization arguments of ego agent such as position, velocity, etc.
    :param x_axis: simulation environment limitation in x-direction.
    :param y_axis: simulation environment limitation in y-direction.
    :param dt: simulation time-step [s].
    :param scene_name: configuration name of initialized environment (for logging purposes only).
    """

    Ghost = namedtuple("Ghost", "agent weight id")

    def __init__(
        self,
        ego_type: Agent.__class__ = None,
        ego_kwargs: Dict[str, Any] = None,
        x_axis: Tuple[float, float] = sim_x_axis_default,
        y_axis: Tuple[float, float] = sim_y_axis_default,
        dt: float = sim_dt_default,
        scene_name: str = "unknown"
    ):
        assert x_axis[0] < x_axis[1], "x axis must be in form (x_min, x_max)"
        assert y_axis[0] < y_axis[1], "y axis must be in form (y_min, y_max)"
        assert dt > 0.0, "time-step must be larger than 0"

        self._ego = ego_type(**ego_kwargs, identifier="ego") if ego_type is not None else None
        self._ado_ghosts = []
        self._num_ado_modes = 0
        self._ado_ids = []

        self._x_axis = x_axis
        self._y_axis = y_axis
        self._dt = dt
        self._sim_time = 0
        self._scene_name = scene_name

    def step(self, ego_control: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Run simulation step (time-step = dt). Update state and history of ados and ego. Also reset simulation time
        to sim_time_new = sim_time + dt. The difference to predict() is two-fold: Firstly, step() is only going forward
        one time-step at a time, not in general `t_horizon` steps, secondly, step() changes the actual agent states
        in the simulation while predict() copies all agents and changes the states of these copies (so the actual
        agent states remain unchanged).

        :param ego_control: planned ego control input for current time step (1, 2).
        :returns: ado_states (num_ados, num_modes, 1, 5), ego_next_state (5) in next time step.
        """
        assert check_ego_controls(ego_control, t_horizon=1)
        self._sim_time = self._sim_time + self.dt

        # Unroll future ego trajectory, which is surely deterministic and certain due to the deterministic dynamics
        # assumption. Update ego based on the first action of the input ego policy.
        self._ego.update(ego_control.flatten(), dt=self.dt)
        logging.info(f"sim {self.name} step @t={self.sim_time} [ego]: action={ego_control.tolist()}")
        logging.info(f"sim {self.name} step @t={self.sim_time} [ego_{self._ego.id}]: state={self.ego.state.tolist()}")

        # Predict the next step in the environment by forward simulation.
        _, ado_controls, weights = self.predict_w_controls(controls=ego_control, return_more=True)

        # Update ados by forward simulate them and determining their most likely policies. Therefore predict the
        # ado states at the next time step as well as the probabilities (weights) of them occurring. Then sample one
        # mode (given these weights) and update the ados as that sampled mode.
        # The base state should be the same between all modes, therefore update all mode states according to the
        # one sampled mode policy.
        weights = weights / torch.sum(weights, dim=1)[:, np.newaxis]
        ado_states = torch.zeros((self.num_ados, 1, 1, 5))  # deterministic update (!)
        sampled_modes = {}
        for ado_id in self.ado_ids:
            i_ado = self.index_ado_id(ado_id=ado_id)
            assert weights[i_ado, :].numel() == self.num_modes
            sampled_modes[ado_id] = np.random.choice(range(self.num_modes), p=weights[i_ado, :])

        # Now update the internal ghost representations accordingly, every ghost originating from the ado should now
        # be "synchronized", i.e. have the same current state.
        for j in range(self.num_ghosts):
            ado_id, _ = self.split_ghost_id(ghost_id=self.ghosts[j].id)
            i_ado = self.index_ado_id(ado_id=ado_id)
            self._ado_ghosts[j].agent.update(action=ado_controls[i_ado, sampled_modes[ado_id], 0, :], dt=self.dt)
            ado_states[i_ado, :, :, :] = self.ghosts[j].agent.state_with_time  # TODO: repetitive !
            logging.info(f"sim {self.name} step @t={self.sim_time} [ado_{ado_id}]: state={ado_states[i_ado].tolist()}")

        # Detach agents from graph in order to keep independence between subsequent runs.
        self.detach()
        return ado_states.detach(), self.ego.state_with_time.detach()  # otherwise no scene independence (!)

    def step_reset(self, ego_state_next: Union[torch.Tensor, None], ado_states_next: Union[torch.Tensor, None]):
        """Run simulation step (time-step = dt). Instead of predicting the behaviour of every agent in the scene, it
        is given as an input and the agents are merely updated. All the ghosts (modes of an ado) will collapse to the
        same given state, since the update is deterministic.

        :param ego_state_next: ego state for next time step (5).
        :param ado_states_next: ado states for next time step (num_ados, num_modes, 1, 5).
        """
        self._sim_time = self._sim_time + self.dt

        # Reset ego agent (if there is an ego in the scene), otherwise just do not reset it.
        if ego_state_next is not None:
            assert check_state(ego_state_next, enforce_temporal=True)
            self._ego.reset(state=ego_state_next, history=None)  # new state is appended

        # Reset ado agents, each mode similarly, if `ado_states_next` is None just do not reset them. When resetting
        # with `history=None` the new state is appended automatically.
        if ado_states_next is not None:
            assert check_trajectories(ado_states_next, ados=self.num_ados, t_horizon=1, modes=1)
            for j in range(self.num_ghosts):
                i_ado, _ = self.index_ghost_id(ghost_id=self.ghosts[j].id)
                self._ado_ghosts[j].agent.reset(ado_states_next[i_ado, 0, 0, :], history=None)

        # Detach agents from graph in order to keep independence between subsequent runs.
        self.detach()

    ###########################################################################
    # Prediction ##############################################################
    ###########################################################################
    def predict_w_controls(self, controls: torch.Tensor, return_more: bool = False, **graph_kwargs) -> torch.Tensor:
        """Predict the environments future for the given time horizon (discrete time).
        The internal prediction model is dependent on the exact implementation of the internal interaction model
        between the ados with each other and between the ados and the ego. The implementation therefore is specific
        to each child-class.

        :param controls: ego control input (pred_horizon, 2).
        :param return_more: return the system inputs (at every time -> trajectory) and probabilities of each mode.
        :return: predicted trajectories for ados in the scene (either one or multiple for each ado).
        """
        assert self.ego is not None
        assert check_ego_controls(controls)
        ego_trajectory = self.ego.unroll_trajectory(controls=controls, dt=self.dt)
        graphs = self.build_connected_graph(ego_trajectory, ego_grad=False, **graph_kwargs)
        return self.transcribe_graph(graphs, t_horizon=controls.shape[0] + 1, returns=return_more)

    def predict_w_trajectory(self, trajectory: torch.Tensor, return_more: bool = False, **graph_kwargs) -> torch.Tensor:
        """Predict the environments future for the given time horizon (discrete time).
        The internal prediction model is dependent on the exact implementation of the internal interaction model
        between the ados with each other and between the ados and the ego. The implementation therefore is specific
        to each child-class.

        :param trajectory: ego trajectory (pred_horizon, 4).
        :param return_more: return the system inputs (at every time -> trajectory) and probabilities of each mode.
        :return: predicted trajectories for ados in the scene (either one or multiple for each ado).
        """
        assert self.ego is not None
        assert check_ego_trajectory(ego_trajectory=trajectory, pos_and_vel_only=True)
        graphs = self.build_connected_graph(ego_trajectory=trajectory, ego_grad=False, **graph_kwargs)
        return self.transcribe_graph(graphs, t_horizon=trajectory.shape[0], returns=return_more)

    def predict_wo_ego(self, t_horizon: int, return_more: bool = False, **graph_kwargs) -> torch.Tensor:
        """Predict the environments future for the given time horizon (discrete time).
        The internal prediction model is dependent on the exact implementation of the internal interaction model
        between the ados while ignoring the ego.

        :param t_horizon: prediction horizon, number of discrete time-steps.
        :param return_more: return the system inputs (at every time -> trajectory) and probabilities of each mode.
        :return: predicted trajectories for ados in the scene (either one or multiple for each ado).
        """
        graphs = self.build_connected_graph_wo_ego(t_horizon=t_horizon, **graph_kwargs)
        return self.transcribe_graph(graphs, t_horizon=t_horizon, returns=return_more)

    ###########################################################################
    # Scene ###################################################################
    ###########################################################################
    def states(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return current states of ego and ado agents in the scene. Since the current state is known for every
        ado the states are deterministic and uni-modal. States are returned as vector including temporal dimension.

        :returns: ego state vector including temporal dimension (5).
        :returns: ado state vectors including temporal dimension (num_ados, 5).
        """
        ado_states = torch.zeros((self.num_ados, self.num_modes, 1, 5))
        for ghost in self.ghosts:
            i_ado, i_mode = self.index_ghost_id(ghost_id=ghost.id)
            ado_states[i_ado, i_mode, 0, :] = ghost.agent.state_with_time
        ego_state = self.ego.state_with_time if self.ego is not None else None
        return ego_state, ado_states

    def add_ado(self, num_modes: int = 1, weights: List[float] = None, arg_list: List[Dict] = None, **ado_kwargs):
        """Add (multi-modal) ado (i.e. non-robot) agent to simulation.
        While the ego is added to the simulation during initialization, the ado agents have to be added afterwards,
        individually. To do so for each mode an agent is initialized using the passed initialization arguments and
        appended to the internal list of ghosts, while staying assignable to the original ado agent by id, i.e.
        ghost_id = ado_id + mode_index.
        Thereby the ghosts are sorted with decreasing level of importance, i.e. decreasing weight, so that the first
        ghost in the list of added ghosts for this agent always is the most important one.

        :param num_modes: number of modes of multi-modal ado agent (>=1).
        :param weights: mode weight vector, default = uniform distribution.
        :param arg_list: initialization arguments for each mode.
        """
        assert "type" in ado_kwargs.keys() and type(ado_kwargs["type"]) == Agent.__class__
        ado = ado_kwargs["type"](**ado_kwargs)
        self._ado_ids.append(ado.id)

        # Append ado to internal list of ados and rebuilt the graph (could be also extended but small computational
        # to actually rebuild it).
        assert self._x_axis[0] <= ado.position[0] <= self._x_axis[1], "ado x position must be in scene"
        assert self._y_axis[0] <= ado.position[1] <= self._y_axis[1], "ado y position must be in scene"
        if self._num_ado_modes == 0:
            self._num_ado_modes = num_modes
        assert num_modes == self.num_modes  # all ados should have same number of modes

        # Append the created ado for every mode.
        arg_list = arg_list if arg_list is not None else [dict()] * num_modes
        weights = weights if weights is not None else (torch.ones(num_modes) / num_modes).tolist()
        index_sorted = list(reversed(np.argsort(weights)))  # per default in increasing order, but we want decreasing
        arg_list = [arg_list[k] for k in index_sorted]
        weights = [weights[k] for k in index_sorted]
        assert len(arg_list) == len(weights) == num_modes

        for i in range(num_modes):
            ado = deepcopy(ado)
            gid = self.build_ghost_id(ado_id=ado.id, mode_index=i)
            self._ado_ghosts.append(self.Ghost(ado, weight=weights[i], id=gid, **arg_list[i]))  # required to be general

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

    def index_ghost_id(self, ghost_id: str) -> Tuple[int, int]:
        ado_id, mode_index = self.split_ghost_id(ghost_id)
        return self.ado_ids.index(ado_id), mode_index

    ###########################################################################
    # Simulation graph ########################################################
    ###########################################################################
    def build_connected_graph(self, trajectory: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Build differentiable graph for predictions over multiple time-steps. For the sake of differentiability
        the computation for the nth time-step cannot be done iteratively, i.e. by determining the current states and
        using the resulting values for computing the next time-step's results in a Markovian manner. Instead the whole
        graph (which is the whole computation) has to be built over n time-steps and evaluated at once by forward pass.

        For building the graph the graphs for each single time-step is built independently while being connected
        using the outputs of the previous time-step and an input for the current time-step. This is quite heavy in
        terms of computational effort and space, however end-to-end-differentiable.
        """
        assert check_ego_trajectory(trajectory, pos_and_vel_only=True)
        return self._build_connected_graph(t_horizon=trajectory.shape[0], trajectory=trajectory, **kwargs)

    @abstractmethod
    def _build_connected_graph(self, t_horizon: int, trajectory: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def build_connected_graph_wo_ego(self, t_horizon: int, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def write_state_to_graph(self, ego_state: torch.Tensor = None, **graph_kwargs) -> Dict[str, torch.Tensor]:
        k = dict_value_or_default(graph_kwargs, key="k", default=0)
        ado_grad = dict_value_or_default(graph_kwargs, key="ado_grad", default=False)
        ego_grad = dict_value_or_default(graph_kwargs, key="ego_grad", default=True)
        graph = {}

        if ego_state is not None:
            graph[f"ego_{k}_position"] = ego_state[0:2]
            graph[f"ego_{k}_velocity"] = ego_state[2:4]

            if ego_grad and not graph[f"ego_{k}_position"].requires_grad:  # if require_grad has been set already ...
                graph[f"ego_{k}_position"].requires_grad = True
                graph[f"ego_{k}_velocity"].requires_grad = True

        for ghost in self.ghosts:
            graph[f"{ghost.id}_{k}_position"] = ghost.agent.position
            graph[f"{ghost.id}_{k}_velocity"] = ghost.agent.velocity
            if ado_grad and graph[f"{ghost.id}_{k}_position"].requires_grad is not True:
                graph[f"{ghost.id}_{k}_position"].requires_grad = True
                graph[f"{ghost.id}_{k}_velocity"].requires_grad = True

        return graph

    def transcribe_graph(self, graph: Dict[str, torch.Tensor], t_horizon: int, returns: bool = False):
        """Remodel simulation outputs, as they are all stored in the simulation graph. """
        controls = torch.zeros((self.num_ados, self.num_modes, t_horizon - 1, 2))
        trajectories = torch.zeros((self.num_ados, self.num_modes, t_horizon, 5))
        weights = torch.zeros((self.num_ados, self.num_modes))

        for j, ghost in enumerate(self.ghosts):
            i_ado, i_mode = self.index_ghost_id(ghost_id=ghost.id)
            for k in range(t_horizon):
                trajectories[i_ado, i_mode, k, 0:2] = graph[f"{ghost.id}_{k}_position"]
                trajectories[i_ado, i_mode, k, 2:4] = graph[f"{ghost.id}_{k}_velocity"]
                trajectories[i_ado, i_mode, k, -1] = self.sim_time + self.dt * k
                if k < t_horizon - 1:
                    controls[i_ado, i_mode, k, :] = graph[f"{ghost.id}_{k}_control"]
            weights[i_ado, i_mode] = ghost.weight

        assert check_controls(controls, num_ados=self.num_ados, num_modes=self.num_modes, t_horizon=t_horizon - 1)
        assert check_weights(weights, num_ados=self.num_ados, num_modes=self.num_modes)
        assert check_trajectories(trajectories, self.num_ados, t_horizon=t_horizon, modes=self.num_modes)
        return trajectories if not returns else (trajectories, controls, weights)

    def detach(self):
        """Detach all internal agents (ego and all ado ghosts) from computation graph. This is sometimes required to
        completely separate subsequent computations in PyTorch."""
        self._ego.detach()
        for m in range(self.num_ghosts):
            self.ghosts[m].agent.detach()

    ###########################################################################
    # Operators ###############################################################
    ###########################################################################
    def copy(self):
        """Create copy of environment. However just using deepcopy is not supported for tensors that are not detached
        from the PyTorch computation graph. Therefore re-initialize the objects such as the agents in the environment
        and reset their state to the internal current state.
        """
        # Create environment copy of internal class, pass simulation parameters such as the forward integration
        # time-step and initialize ego agent.
        ego_type = None
        ego_kwargs = None
        if self.ego is not None:
            ego_type = self.ego.__class__
            position = self.ego.position
            velocity = self.ego.velocity
            history = self.ego.history
            identifier = self.ego.id
            ego_kwargs = {"position": position, "velocity": velocity, "history": history, "identifier": identifier}

        (x_axis, y_axis), dt, name = self.axes, self.dt, self.scene_name
        env_copy = self.__class__(ego_type, ego_kwargs, x_axis=x_axis, y_axis=y_axis, dt=dt, scene_name=name)

        # Add internal ado agents to newly created environment.
        for i in range(self.num_ados):
            ghosts_ado = self.ghosts_by_ado_index(ado_index=i)
            ado_id, _ = self.split_ghost_id(ghost_id=ghosts_ado[0].id)
            env_copy.add_ado(
                position=ghosts_ado[0].agent.position,  # same over all ghosts of same ado
                velocity=ghosts_ado[0].agent.velocity,  # same over all ghosts of same ado
                history=ghosts_ado[0].agent.history,  # same over all ghosts of same ado
                weights=[ghost.weight for ghost in ghosts_ado],
                num_modes=self.num_modes,
                identifier=self.split_ghost_id(ghost_id=ghosts_ado[0].id)[0]
            )
        return env_copy

    def same_initial_conditions(self, other):
        """Similar to __eq__() function, but not enforcing parameters of simulation to be completely equivalent,
        merely enforcing the initial conditions to be equal, such as states of agents in scene. Hence, all prediction
        depending parameters, namely the number of modes or agent's parameters dont have to be equal.
        """
        is_equal = True
        is_equal = is_equal and self.dt == other.dt
        is_equal = is_equal and self.num_ados == other.num_ados
        is_equal = is_equal and self.ego == other.ego
        ghosts_equal = [True] * self.num_ados
        ghosts_imp = self.ados_most_important_mode()
        other_imp = other.ados_most_important_mode()
        for i in range(self.num_ados):
            ghosts_equal[i] = ghosts_equal[i] and ghosts_imp[i].agent == other_imp[i].agent
        is_equal = is_equal and all(ghosts_equal)
        return is_equal

    ###########################################################################
    # Ado properties ##########################################################
    ###########################################################################
    @property
    def ado_colors(self) -> List[List[float]]:
        ados = self.ados_most_important_mode()
        return [ado.agent.color for ado in ados]

    @property
    def ado_ids(self) -> List[str]:
        assert len(self.ghosts) == len(self._ado_ids) * self._num_ado_modes
        return self._ado_ids

    ###########################################################################
    # Ghost properties ########################################################
    # Per default the ghosts (i.e. the multimodal representations of the ados #
    # are the ados themselves, as the default case is uni-modal. ##############
    ###########################################################################
    @property
    def ghosts(self) -> List[Ghost]:
        return self._ado_ghosts

    @property
    def num_ados(self) -> int:
        return len(self.ado_ids)

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
    def sim_time(self) -> float:
        return round(self._sim_time, 2)

    @property
    def axes(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return self._x_axis, self._y_axis

    @property
    def simulation_name(self) -> str:
        return self.__class__.__name__.lower()

    @property
    def scene_name(self) -> str:
        return self._scene_name

    @property
    def name(self) -> str:
        return self.simulation_name + "_" + self._scene_name
