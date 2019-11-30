from abc import abstractmethod
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

        self._graph = None
        self._build_graph()

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

    def step(self, ego_trajectory: np.ndarray = None):
        """Run simulation step (time-step = dt). Update state and history of ados and ego. Also reset simulation time
        to sim_time_new = sim_time + dt.
        :param ego_trajectory: planned ego trajectory (in case of dependence in behaviour between ado and ego).
        """
        logging.debug(f"Simulation update step: {self.sim_time:.2f} -> {self.sim_time + self.dt:.2f}")
        _, policies = self.predict(t_horizon=1, ego_trajectory=ego_trajectory, return_policies=True)
        for i, ado in enumerate(self._ados):
            ado.update(policies[0, i], dt=self.dt)
        self._sim_time = self.sim_time + self.dt

    def _add_ado(self, ado_type: Agent.__class__ = None, **ado_kwargs):
        ado = ado_type(**ado_kwargs)
        assert self._x_axis[0] < ado.position[0] < self._x_axis[1], "ado x position must be in scene"
        assert self._y_axis[0] < ado.position[1] < self._y_axis[1], "ado y position must be in scene"

        # Append ado to internal list of ados and rebuilt the graph (could be also extended but small computational
        # to actually rebuild it).
        self._ados.append(ado)
        self._build_graph()

    @abstractmethod
    def _build_graph(self):
        """The simulation should be defined as differentiable graph. To make it accessible (e.g. for the planner)
        and to save computational effort the graph is pre-built. The exact definition of the graph depends on the
        actual simulation and is therefore defined in the child class."""
        pass

    @property
    def ados(self) -> List[Agent]:
        return self._ados

    @property
    def ego(self) -> Union[Agent, None]:
        return self._ego

    @property
    def graph(self) -> torch.Tensor:
        return self._graph

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def sim_time(self) -> float:
        return round(self._sim_time, 2)

    @property
    def axes(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return self._x_axis, self._y_axis
