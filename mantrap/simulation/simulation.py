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
from mantrap.utility.shaping import check_state, check_trajectories, check_controls, check_weights


class GraphBasedSimulation:

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

        :param ego_control: planned ego control input for current time step (2).
        :return ado_states (num_ados, num_modes, 1, 5), ego_next_state (5) in next time step.
        """
        self._sim_time = self._sim_time + self.dt

        # Unroll future ego trajectory, which is surely deterministic and certain due to the deterministic dynamics
        # assumption. Update ego based on the first action of the input ego policy.
        self._ego.update(ego_control, dt=self.dt)
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
        for ado_id in self.ado_ids:   # TODO: enforce same order of ado_id and weights
            i_ado = self.index_ado_id(ado_id=ado_id)
            assert weights[i_ado, :].numel() == self.num_ado_modes
            sampled_modes[ado_id] = np.random.choice(range(self.num_ado_modes), p=weights[i_ado, :])

        # Now update the internal ghost representations accordingly, every ghost originating from the ado should now
        # be "synchronized", i.e. have the same current state.
        for j in range(self.num_ado_ghosts):
            ado_id, _ = self.split_ghost_id(ghost_id=self.ado_ghosts[j].id)
            i_ado = self.index_ado_id(ado_id=ado_id)
            self._ado_ghosts[j].agent.update(action=ado_controls[i_ado, sampled_modes[ado_id], 0, :], dt=self.dt)
            ado_states[i_ado, :, :, :] = self.ado_ghosts[j].agent.state_with_time  # TODO: repetitive !
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
            for j in range(self.num_ado_ghosts):
                i_ado, _ = self.index_ghost_id(ghost_id=self.ado_ghosts[j].id)
                self._ado_ghosts[j].agent.reset(ado_states_next[i_ado, 0, 0, :], history=None)

    ###########################################################################
    # Prediction ##############################################################
    ###########################################################################
    @abstractmethod
    def predict_w_controls(self, controls: torch.Tensor, return_more: bool = False, **graph_kwargs) -> torch.Tensor:
        """Predict the environments future for the given time horizon (discrete time).
        The internal prediction model is dependent on the exact implementation of the internal interaction model
        between the ados with each other and between the ados and the ego. The implementation therefore is specific
        to each child-class.

        :param controls: ego control input (pred_horizon, 2).
        :param return_more: return the system inputs (at every time -> trajectory) and probabilities of each mode.
        :return: predicted trajectories for ados in the scene (either one or multiple for each ado).
        """
        raise NotImplementedError

    @abstractmethod
    def predict_w_trajectory(self, trajectory: torch.Tensor, return_more: bool = False,
                             **graph_kwargs) -> torch.Tensor:
        """Predict the environments future for the given time horizon (discrete time).
        The internal prediction model is dependent on the exact implementation of the internal interaction model
        between the ados with each other and between the ados and the ego. The implementation therefore is specific
        to each child-class.

        :param trajectory: ego trajectory (pred_horizon, 4).
        :param return_more: return the system inputs (at every time -> trajectory) and probabilities of each mode.
        :return: predicted trajectories for ados in the scene (either one or multiple for each ado).
        """
        raise NotImplementedError

    @abstractmethod
    def predict_wo_ego(self, t_horizon: int, return_more: bool = False, **graph_kwargs) -> torch.Tensor:
        """Predict the environments future for the given time horizon (discrete time).
        The internal prediction model is dependent on the exact implementation of the internal interaction model
        between the ados while ignoring the ego.

        :param t_horizon: prediction horizon, number of discrete time-steps.
        :param return_more: return the system inputs (at every time -> trajectory) and probabilities of each mode.
        :return: predicted trajectories for ados in the scene (either one or multiple for each ado).
        """
        pass

    ###########################################################################
    # Scene ###################################################################
    ###########################################################################
    def states(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return current states of ego and ado agents in the scene. Since the current state is known for every
        ado the states are deterministic and uni-modal. States are returned as vector including temporal dimension.
        """
        ado_states = torch.zeros((self.num_ados, self.num_ado_modes, 1, 5))
        for ghost in self.ado_ghosts:
            i_ado, i_mode = self.index_ghost_id(ghost_id=ghost.id)
            ado_states[i_ado, i_mode, 0, :] = ghost.agent.state
        return self.ego.state_with_time, ado_states

    def add_ado(self, num_modes: int = 1, weights: List[float] = None, arg_list: List[Dict] = None, **ado_kwargs):
        assert "type" in ado_kwargs.keys() and type(ado_kwargs["type"]) == Agent.__class__, "ado type required"
        ado = ado_kwargs["type"](**ado_kwargs)
        self._ado_ids.append(ado.id)

        # Append ado to internal list of ados and rebuilt the graph (could be also extended but small computational
        # to actually rebuild it).
        assert self._x_axis[0] <= ado.position[0] <= self._x_axis[1], "ado x position must be in scene"
        assert self._y_axis[0] <= ado.position[1] <= self._y_axis[1], "ado y position must be in scene"
        if self._num_ado_modes == 0:
            self._num_ado_modes = num_modes
        assert num_modes == self.num_ado_modes  # all ados should have same number of modes

        # Append the created ado for every mode.
        arg_list = arg_list if arg_list is not None else [dict()] * num_modes
        weights = weights if weights is not None else (torch.ones(num_modes) / num_modes).tolist()
        assert len(arg_list) == len(weights) == num_modes
        for i in range(num_modes):
            ado = deepcopy(ado)
            gid = self.build_ghost_id(ado_id=ado.id, mode_index=i)
            self._ado_ghosts.append(self.Ghost(ado, weight=weights[i], id=gid, **arg_list[i]))  # required to be general

    def ados_most_important_mode(self) -> List[Ghost]:
        """Return a list of the most important ghosts, i.e. the ones with the highest weight, for each ado."""
        ado_ghost_dict = {}
        for ghost in self.ado_ghosts:
            ado_id = ghost.agent.id
            if ado_id not in ado_ghost_dict.keys() or ado_ghost_dict[ado_id].weight < ghost.weight:
                ado_ghost_dict[ado_id] = ghost

        assert len(ado_ghost_dict.values()) == self.num_ados
        return list(ado_ghost_dict.values())

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
    @abstractmethod
    def build_connected_graph(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Build differentiable graph for predictions over multiple time-steps. For the sake of differentiability
         the computation for the nth time-step cannot be done iteratively, i.e. by determining the current states and
         using the resulting values for computing the next time-step's results in a Markovian manner. Instead the whole
         graph (which is the whole computation) has to be built over n time-steps and evaluated at once by forward pass.

         For building the graph the graphs for each single time-step is built independently while being connected
         using the outputs of the previous time-step and an input for the current time-step. This is quite heavy in
         terms of computational effort and space, however end-to-end-differentiable.
        """
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

        for ghost in self.ado_ghosts:
            graph[f"{ghost.id}_{k}_position"] = ghost.agent.position
            graph[f"{ghost.id}_{k}_velocity"] = ghost.agent.velocity
            if ado_grad and graph[f"{ghost.id}_{k}_position"].requires_grad is not True:
                graph[f"{ghost.id}_{k}_position"].requires_grad = True
                graph[f"{ghost.id}_{k}_velocity"].requires_grad = True

        return graph

    def transcribe_graph(self, graph: Dict[str, torch.Tensor], t_horizon: int, returns: bool = False):
        """Remodel simulation outputs, as they are all stored in the simulation graph. """
        controls = torch.zeros((self.num_ados, self.num_ado_modes, t_horizon - 1, 2))
        trajectories = torch.zeros((self.num_ados, self.num_ado_modes, t_horizon, 5))
        weights = torch.zeros((self.num_ados, self.num_ado_modes))

        for j, ghost in enumerate(self.ado_ghosts):
            i_ado, i_mode = self.index_ghost_id(ghost_id=ghost.id)
            for k in range(t_horizon):
                trajectories[i_ado, i_mode, k, 0:2] = graph[f"{ghost.id}_{k}_position"]
                trajectories[i_ado, i_mode, k, 2:4] = graph[f"{ghost.id}_{k}_velocity"]
                trajectories[i_ado, i_mode, k, -1] = self.sim_time + self.dt * k
                if k < t_horizon - 1:
                    controls[i_ado, i_mode, k, :] = graph[f"{ghost.id}_{k}_force"]
            weights[i_ado, i_mode] = ghost.weight

        assert check_controls(controls, num_ados=self.num_ados, num_modes=self.num_ado_modes, t_horizon=t_horizon - 1)
        assert check_weights(weights, num_ados=self.num_ados, num_modes=self.num_ado_modes)
        assert check_trajectories(trajectories, self.num_ados, t_horizon=t_horizon, modes=self.num_ado_modes)
        return trajectories if not returns else (trajectories, controls, weights)

    def detach(self):
        self._ego.detach()
        for m in range(self.num_ado_ghosts):
            self.ado_ghosts[m].agent.detach()

    ###########################################################################
    # Operators ###############################################################
    ###########################################################################
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
        assert len(self.ado_ghosts) == len(self._ado_ids) * self._num_ado_modes
        return self._ado_ids

    ###########################################################################
    # Ghost properties ########################################################
    # Per default the ghosts (i.e. the multimodal representations of the ados #
    # are the ados themselves, as the default case is uni-modal. ##############
    ###########################################################################
    @property
    def ado_ghosts(self) -> List[Ghost]:
        return self._ado_ghosts

    @property
    def num_ados(self) -> int:
        return len(self.ado_ids)

    @property
    def num_ado_ghosts(self) -> int:
        return len(self._ado_ghosts)

    @property
    def num_ado_modes(self) -> int:
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
    def name(self) -> str:
        return self.simulation_name + "_" + self._scene_name
