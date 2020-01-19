from abc import abstractmethod
import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from mantrap.agents.agent import Agent
from mantrap.constants import sim_x_axis_default, sim_y_axis_default, sim_dt_default


class Simulation:
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

        self._ego = ego_type(**ego_kwargs) if ego_type is not None else None
        self._ados = []

        self._x_axis = x_axis
        self._y_axis = y_axis
        self._dt = dt
        self._sim_time = 0

    @abstractmethod
    def predict(
        self, t_horizon: int, ego_trajectory: np.ndarray = None, return_policies: bool = False,
    ) -> np.ndarray:
        """Predict the environments future for the given time horizon (discrete time).
        The internal prediction model is dependent on the exact implementation of the internal interaction model
        between the ados with each other and between the ados and the ego. The implementation therefore is specific
        to each child-class. If the ego_trajectory is set to none, the interaction between ego and any ado is not
        taken into account, instead just between ados.
        Dependent on whether the prediction is deterministic or probabilistic the output can vary between each child-
        class, by setting the prediction_t. However the output should be a vector of predictions, one for each ado.
        :param t_horizon: prediction horizon (number of time-steps of length dt).
        :param ego_trajectory: planned ego trajectory (in case of dependence in behaviour between ado and ego).
        :param return_policies: return the actual system inputs (at every time to get trajectory).
        :return: predicted trajectories for ados in the scene (either one or multiple for each ado).
        """
        pass

    def step(self, ego_policy: np.ndarray = None) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        """Run simulation step (time-step = dt). Update state and history of ados and ego. Also reset simulation time
        to sim_time_new = sim_time + dt. The difference to predict() is two-fold: Firstly, step() is only going forward
        one time-step at a time, not in general `t_horizon` steps, secondly, step() changes the actual agent states
        in the simulation while predict() copies all agents and changes the states of these copies (so the actual
        agent states remain unchanged).
        :param ego_policy: planned ego policy (in case of dependence in behaviour between ado and ego).
        :return ado_states (num_ados, num_modes, 6), ego_next_state (6) in next time step.
        """
        self._sim_time = self._sim_time + self.dt

        # Unroll future ego trajectory, which is surely deterministic and certain due to the deterministic dynamics
        # assumption. Update ego based on the first action of the input ego policy.
        ego_trajectory, ego_next_state = None, None
        if ego_policy is not None:
            ego_policy = np.expand_dims(ego_policy, axis=0) if len(ego_policy.shape) == 1 else ego_policy
            ego_trajectory = self.ego.unroll_trajectory(policy=ego_policy, dt=self.dt)
            ego_trajectory[:, -1] = self._sim_time  # correct time
            ego_next_state = ego_trajectory[1, :]
            self._ego.update(ego_policy[0, :], dt=self.dt)
            logging.info(f"simulation @t={self.sim_time} [ego_{self._ego.id}]: policy={ego_policy}")

        # # Update ados by forward simulate them and determining their most likely policies.
        ado_states, policies = self.predict(t_horizon=1, ego_trajectory=ego_trajectory, return_policies=True)
        for i, ado in enumerate(self._ados):
            ado.update(policies[0, i], dt=self.dt)
            logging.info(f"simulation @t={self.sim_time} [ado_{ado.id}]: state={ado_states[i, 0, :, :]}")

        if self._ego is not None:
            logging.info(f"simulation @t={self.sim_time} [ego_{self._ego.id}]: state={ego_next_state}")
        return ado_states, ego_next_state

    def add_ado(self, **ado_kwargs):
        assert "type" in ado_kwargs.keys() and type(ado_kwargs["type"]) == Agent.__class__, "ado type required"
        ado = ado_kwargs["type"](**ado_kwargs)
        assert self._x_axis[0] <= ado.position[0] <= self._x_axis[1], "ado x position must be in scene"
        assert self._y_axis[0] <= ado.position[1] <= self._y_axis[1], "ado y position must be in scene"

        # Append ado to internal list of ados and rebuilt the graph (could be also extended but small computational
        # to actually rebuild it).
        self._ados.append(ado)

    def reset(self, ego_state: np.ndarray, ado_states: np.ndarray, ado_histories: np.ndarray = None):
        """Completely reset the simulation to the given states. Additionally, the ado histories can be resetted,
        if given as None, they will be re-initialized. The other simulation variables remain unchanged.
        :param ego_state: new ego state containing (x, y, theta, vx, vy), (5).
        :param ado_states: new state for each ado containing (x, y, theta, vx, vy), (num_ados, 5), number of ados
                           cannot be changed (!).
        :param ado_histories: new history for each ado containing (x, y, theta, vx, vy, t), (num_ados, 6).
        """
        assert ego_state.size >= 5, "ego state must contain (x, y, theta, vx, vy)"
        assert len(ado_states.shape) == 2, "ado states must be of shape (num_ados, 5)"
        assert ado_states.shape[0] == self.num_ados, "number of ados and ado_states must match"
        assert ado_states.shape[1] >= 5, "ado states must contain (x, y, theta, vx, vy)"
        if ado_histories is not None:
            assert ado_histories.shape[0] == self.num_ados, "number of ados and ado_histories must match"
            assert ado_histories.shape[2] == 6, "ado histories must contain (x, y, theta, vx, vy, t)"

        self._ego.reset(position=ego_state[0:2], velocity=ego_state[3:5], history=None)
        for i in range(ado_states.shape[0]):
            ado_i_history = ado_histories[i, :, :] if ado_histories is not None else None
            self._ados[i].reset(ado_states[i, 0:2], ado_states[i, 3:5], history=ado_i_history)

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
    def ado_colors(self) -> List[np.ndarray]:
        return [ado.color for ado in self._ados]

    @property
    def ado_ids(self) -> List[str]:
        return [ado.id for ado in self._ados]

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
