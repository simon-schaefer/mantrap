from abc import abstractmethod
import copy
import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

import mantrap.constants
from mantrap.agents.agent import Agent


class Simulation:
    def __init__(
        self,
        ego_type: Agent.__class__ = None,
        ego_kwargs: Dict[str, Any] = None,
        x_axis: Tuple[float, float] = mantrap.constants.sim_x_axis_default,
        y_axis: Tuple[float, float] = mantrap.constants.sim_y_axis_default,
        dt: float = mantrap.constants.sim_dt_default,
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
        self,
        t_horizon: int = mantrap.constants.t_horizon_default,
        ego_trajectory: np.ndarray = None,
        return_policies: bool = False,
    ) -> np.ndarray:
        """Predict the environments future for the given time horizon (discrete time).
        The internal prediction model is dependent on the exact implementation of the internal interaction model
        between the ados with each other and between the ados and the ego. The implementation therefore is specific
        to each child-class.
        Dependent on whether the prediction is deterministic or probabilistic the output can vary between each child-
        class, by setting the prediction_t. However the output should be a vector of predictions, one for each ado.
        :param t_horizon: prediction horizon (number of time-steps of length dt).
        :param ego_trajectory: planned ego trajectory (in case of dependence in behaviour between ado and ego).
        :param return_policies: return the actual system inputs (at every time to get trajectory).
        :return: predicted trajectories for ados in the scene (either one or multiple for each ado).
        """
        pass

    def step(self, ego_policy: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Run simulation step (time-step = dt). Update state and history of ados and ego. Also reset simulation time
        to sim_time_new = sim_time + dt.
        :param ego_policy: planned ego policy (in case of dependence in behaviour between ado and ego).
        :return ado_states (num_ados, num_modes, 6), ego_state (6) in next time step.
        """
        self._sim_time = self.sim_time + self.dt

        # Unroll future ego trajectory, which is surely deterministic and certain due to the deterministic dynamics
        # assumption.
        ego_trajectory, ego_state = None, None
        if ego_policy is not None:
            ego_trajectory = self._ego.unroll_trajectory(ego_policy, dt=self.dt)
            ego_trajectory[:, -1] = self.sim_time  # correct time
            ego_state = ego_trajectory[1, :]

        # Update ados by forward simulate them and determining their most likely policies.
        ado_states, policies = self.predict(t_horizon=1, ego_trajectory=ego_trajectory, return_policies=True)
        for i, ado in enumerate(self._ados):
            ado.update(policies[0, i], dt=self.dt)
            logging.debug(f"simulation @t={self.sim_time} [ado_{ado.id}]: {ado_states[i, :, :]}")

        # Update ego based on the first action of the input ego policy.
        if ego_policy is not None:
            ego_policy = np.expand_dims(ego_policy, axis=0) if len(ego_policy.shape) == 1 else ego_policy
            self._ego.update(ego_policy[0, :], dt=self.dt)
            logging.debug(f"simulation @t={self.sim_time} [ego_{self._ego.id}]: policy={ego_policy}")

        if self._ego is not None:
            logging.debug(f"simulation @t={self.sim_time} [ego_{self._ego.id}]: {ego_state}")
        return ado_states, ego_state

    def _add_ado(self, ado_type: Agent.__class__ = None, **ado_kwargs):
        ado = ado_type(**ado_kwargs)
        assert self._x_axis[0] <= ado.position[0] <= self._x_axis[1], "ado x position must be in scene"
        assert self._y_axis[0] <= ado.position[1] <= self._y_axis[1], "ado y position must be in scene"

        # Append ado to internal list of ados and rebuilt the graph (could be also extended but small computational
        # to actually rebuild it).
        self._ados.append(ado)

    @abstractmethod
    def build_graph(self, ados: List[Agent] = None, ego_state: np.ndarray = None) -> Dict[str, torch.Tensor]:
        """The simulation should be defined as differentiable graph. To make it accessible (e.g. for the planner)
        and to save computational effort the graph is pre-built. The exact definition of the graph depends on the
        actual simulation and is therefore defined in the child class.
        If the list of ados is set to None the internal ado state is used for building the graph. If the ego_state
        is None, the interaction between the ego and the environment is ignored."""
        pass

    def graph_check(self, graph: Dict[str, torch.Tensor]) -> bool:
        """Check healthiness of graph by looking for specific keys in the graph that are required."""
        is_okay = True
        is_okay = is_okay and all([req_key in graph.keys() for req_key in []])
        is_okay = is_okay and all([f"{ado.id}_force_norm" for ado in self._ados])
        return is_okay

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


class ForcesBasedSimulation(Simulation):
    def __init__(
        self,
        ego_type: Agent.__class__ = None,
        ego_kwargs: Dict[str, Any] = None,
        x_axis: Tuple[float, float] = mantrap.constants.sim_x_axis_default,
        y_axis: Tuple[float, float] = mantrap.constants.sim_y_axis_default,
        dt: float = mantrap.constants.sim_dt_default,
    ):
        super(ForcesBasedSimulation, self).__init__(ego_type, ego_kwargs, x_axis=x_axis, y_axis=y_axis, dt=dt)

    def predict(
        self,
        t_horizon: int = mantrap.constants.t_horizon_default,
        ego_trajectory: np.ndarray = None,
        return_policies: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if ego_trajectory is not None:
            assert ego_trajectory.shape[0] >= t_horizon, "t_horizon must match length of ego trajectory"

        num_ados = len(self._ados)
        forces = np.zeros((t_horizon, num_ados, 2))

        # The social forces model predicts from one time-step to another, therefore the ados are actually updated in
        # each time step, in order to predict the next time-step. To not change the initial state, hence, the ados
        # vector is copied.
        ados_sim = copy.deepcopy(self.ados)
        for t in range(t_horizon):
            # Build graph based on simulation ados. Build it once and update it every time is surprisingly difficult
            # since the gradients/computations are not updated when one of the leafs is updated, resulting in the
            # same output. However the computational effort of building the graph is negligible (about 1 ms for
            # 2 agents on Mac Pro 2018).
            ego_state = ego_trajectory[t, :] if ego_trajectory is not None else None
            graph_at_t = self.build_graph(ados_sim, ego_state=ego_state)

            # Evaluate graph.
            for i in range(num_ados):
                forces[t, i, :] = graph_at_t[f"{ados_sim[i].id}_force"].detach().numpy()
                ados_sim[i].update(forces[t, i, :], self.dt)  # assuming m = 1 kg

        # Collect histories of simulated ados (last t_horizon steps are equal to future trajectories).
        trajectories = np.asarray([ado.history[-t_horizon:, :] for ado in ados_sim])
        assert trajectories.shape[0] == num_ados, "each ado must be assigned to trajectory"

        if not return_policies:
            return trajectories
        else:
            return trajectories, forces

    @abstractmethod
    def build_graph(self, ados: List[Agent] = None, ego_state: np.ndarray = None) -> Dict[str, torch.Tensor]:
        pass
