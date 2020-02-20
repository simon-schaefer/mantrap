from abc import abstractmethod
from collections import namedtuple
import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from mantrap.agents.agent import Agent
from mantrap.constants import sim_x_axis_default, sim_y_axis_default, sim_dt_default


class Simulation:

    Ghost = namedtuple("Ghost", "agent weight id")

    def __init__(
        self,
        ego_type: Agent.__class__ = None,
        ego_kwargs: Dict[str, Any] = None,
        x_axis: Tuple[float, float] = sim_x_axis_default,
        y_axis: Tuple[float, float] = sim_y_axis_default,
        dt: float = sim_dt_default,
    ):
        assert x_axis[0] < x_axis[1], "x axis must be in form (x_min, x_max)"
        assert y_axis[0] < y_axis[1], "y axis must be in form (y_min, y_max)"
        assert dt > 0.0, "time-step must be larger than 0"

        self._ego = ego_type(**ego_kwargs, identifier="ego") if ego_type is not None else None
        self._ados = []

        self._x_axis = x_axis
        self._y_axis = y_axis
        self._dt = dt
        self._sim_time = 0

    @abstractmethod
    def predict(
        self, graph_input: Union[int, torch.Tensor], returns: bool = False, **graph_kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Predict the environments future for the given time horizon (discrete time).
        The internal prediction model is dependent on the exact implementation of the internal interaction model
        between the ados with each other and between the ados and the ego. The implementation therefore is specific
        to each child-class. If the ego_trajectory is set to none, the interaction between ego and any ado is not
        taken into account, instead just between ados.
        Dependent on whether the prediction is deterministic or probabilistic the output can vary between each child-
        class, by setting the prediction_t. However the output should be a vector of predictions, one for each ado.

        :param graph_input: either ego trajectory or t_horizon.
        :param returns: return the actual system inputs (at every time -> trajectory) and probabilities of each mode.
        :return: predicted trajectories for ados in the scene (either one or multiple for each ado).
        """
        pass

    def step(self, ego_policy: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Run simulation step (time-step = dt). Update state and history of ados and ego. Also reset simulation time
        to sim_time_new = sim_time + dt. The difference to predict() is two-fold: Firstly, step() is only going forward
        one time-step at a time, not in general `t_horizon` steps, secondly, step() changes the actual agent states
        in the simulation while predict() copies all agents and changes the states of these copies (so the actual
        agent states remain unchanged).

        :param ego_policy: planned ego policy (in case of dependence in behaviour between ado and ego).
        :return ado_states (num_ados, num_modes, 5), ego_next_state (5) in next time step.
        """
        self._sim_time = self._sim_time + self.dt

        # Unroll future ego trajectory, which is surely deterministic and certain due to the deterministic dynamics
        # assumption. Update ego based on the first action of the input ego policy.
        ego_policy = ego_policy.unsqueeze(dim=0) if len(ego_policy.shape) == 1 else ego_policy
        self._ego.update(ego_policy[0, :], dt=self.dt)
        logging.info(f"simulation @t={self.sim_time} [ego]: policy={ego_policy}")

        # Predict the next step in the environment by forward simulation.
        ado_states, policies, weights = self.predict(graph_input=self.ego.history[-2:, :], returns=True)
        ado_states = ado_states[:, :, -1, :].unsqueeze(dim=2)

        # Update ados by forward simulate them and determining their most likely policies. Therefore predict the
        # ado states at the next time step as well as the probabilities (weights) of them occurring. Then sample one
        # mode (given these weights) and update the ados as that sampled mode.
        weights = weights / torch.sum(weights, dim=1)[:, np.newaxis]
        for i, ado in enumerate(self._ados):
            sampled_mode = np.random.choice(range(self.num_ado_modes), p=weights[i, :])
            ado.update(policies[i, sampled_mode, 0, :], dt=self.dt)
            logging.info(f"simulation @t={self.sim_time} [ado_{ado.id}]: state={ado_states[i, :]}")

        logging.info(f"simulation @t={self.sim_time} [ego_{self._ego.id}]: state={self.ego.state}")
        return ado_states.detach(), self.ego.state_with_time.detach()  # otherwise no scene independence (!)

    def add_ado(self, **ado_kwargs):
        assert "type" in ado_kwargs.keys() and type(ado_kwargs["type"]) == Agent.__class__, "ado type required"
        ado = ado_kwargs["type"](**ado_kwargs)

        # Append ado to internal list of ados and rebuilt the graph (could be also extended but small computational
        # to actually rebuild it).
        assert self._x_axis[0] <= ado.position[0] <= self._x_axis[1], "ado x position must be in scene"
        assert self._y_axis[0] <= ado.position[1] <= self._y_axis[1], "ado y position must be in scene"
        self._ados.append(ado)

    ###########################################################################
    # Ado properties ##########################################################
    ###########################################################################

    @property
    def ados(self) -> List[Agent]:
        return self._ados

    @property
    def num_ados(self) -> int:
        return len(self._ados)

    @property
    def num_ado_modes(self) -> int:
        return 1

    @property
    def ado_colors(self) -> List[List[float]]:
        return [ado.color for ado in self._ados]

    @property
    def ado_ids(self) -> List[str]:
        return [ado.id for ado in self._ados]

    ###########################################################################
    # Ghost properties ########################################################
    # Per default the ghosts (i.e. the multimodal representations of the ados #
    # are the ados themselves, as the default case is uni-modal. ##############
    ###########################################################################

    @property
    def ado_ghosts_agents(self) -> List[Agent]:
        return self.ados

    @property
    def ado_ghosts(self) -> List[Ghost]:
        return [self.Ghost(ado, weight=torch.ones(1), id=f"{ado.id}_0") for ado in self.ados]

    @property
    def num_ado_ghosts(self) -> int:
        return self.num_ados

    def ghost_to_ado_index(self, ghost_index: int) -> Tuple[int, int]:
        """Ghost of the same "parent" agent are appended to the internal storage of ghosts together, therefore it can
        be backtracked which ghost index belongs to which agent and mode by simple integer division (assuming the same
        number of modes of every ado).
        :return ado index, mode index
        """
        return int(ghost_index / self.num_ado_modes), int(ghost_index % self.num_ado_modes)

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
