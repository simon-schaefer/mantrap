from typing import Any, Dict, List, Tuple, Union

import numpy as np

from murseco.environment import Environment
from murseco.problem.abstract import AbstractProblem


class D2TSProblem(AbstractProblem):
    """The DiscreteTimeDiscreteSpace (D2TS) Problem defines the optimization problem as following, for the time-step
    0 <= k <= N, the discrete space vector x_k, the system dynamics f(x_k, u_k) and the time and spatially dependent
    probability density function of the risk pdf(x_k, k):

    Cost:
        J(x(t)) = sum_{k=0}^{N-1} l(x_k, u_k) + l_f(x_N)
        l(x_k, u_k) = w_x * ||x_Goal - x_k||_2^2 + w_u * ||u||_2^2
        l_f(x_N) = 0 if ||x_G - x_N||^2 < eps, c_Goal otherwise

    Constraints C:
        x_{k+1} = f(x_k, u_k)
        x_0 = x_Start
        ||u_k||_2 < u_max
        sum_{k=0}^N pdf(x_k, k) < risk_{max}

    Optimization:
        min J(x(t)) subject to constraints C
    """

    def __init__(
        self,
        env: Environment,
        x_goal: np.ndarray,
        thorizon: int = 20,
        w_x: float = 1.0,
        w_u: float = 1.0,
        c_goal: float = 1000.0,
        risk_max: float = 0.1,
        u_max: float = 1.0,
        dt: float = 1.0,
        mproc: bool = True,
        grid_resolution: float = 0.01,
        **kwargs,
    ):
        kwargs.update({"name": "problem/d2ts/D2TSProblem"})
        super(D2TSProblem, self).__init__(
            env=env,
            x_goal=x_goal,
            thorizon=thorizon,
            w_x=w_x,
            w_u=w_u,
            c_goal=c_goal,
            risk_max=risk_max,
            u_max=u_max,
            dt=dt,
            mproc=mproc,
            **kwargs,
        )

        # Temporal and spatial discretization.
        assert env.xaxis == env.yaxis, "discretization assumes equal axes for simplification (and nicer graphs)"
        self._grid_resolution = grid_resolution
        self._num_points_axis = int((env.xaxis[1] - env.xaxis[0]) / grid_resolution)
        self._tppdf, self._meshgrid = env.tppdf_grid(thorizon=thorizon, num_points=self._num_points_axis, mproc=mproc)

    @property
    def grid(self) -> Tuple[List[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        return self._tppdf, self._meshgrid

    def cont2discrete(self, cont: np.ndarray) -> Union[np.ndarray, None]:
        """Convert continuous 2D point(s) to discrete grid 2D index(s) given internal parameters."""
        (x_min, x_max), (y_min, y_max) = self._env.xaxis, self._env.yaxis
        if cont is None:
            return None
        elif cont.size == 2:
            assert x_min <= cont[0] <= x_max, "continuous point outside of problem grid (x)"
            assert y_min <= cont[1] <= y_max, "continuous point outside of problem grid (y)"
            x_grid = int(np.round((cont[0] - x_min) / self._grid_resolution))
            y_grid = int(np.round((cont[1] - y_min) / self._grid_resolution))
            return np.array([x_grid, y_grid])
        else:
            assert cont.shape[1] == 2, "stacked points must be two-dimensional"
            assert (cont[:, 0] >= x_min).all() and (cont[:, 0] < x_max).all()
            assert (cont[:, 1] >= y_min).all() and (cont[:, 1] < y_max).all()
            x_grid = np.round((cont[:, 0] - x_min) / self._grid_resolution).astype(int)
            y_grid = np.round((cont[:, 1] - y_min) / self._grid_resolution).astype(int)
            return np.hstack((x_grid, y_grid))

    def discrete2cont(self, discrete: np.ndarray) -> Union[np.ndarray, None]:
        """Convert discrete 2D grid index(s) to continuous 2D points given internal parameters."""
        (x_min, x_max), (y_min, y_max) = self._env.xaxis, self._env.yaxis
        if discrete is None:
            return None
        elif discrete.size == 2:
            assert 0 <= discrete[0] <= self._num_points_axis, "discrete point outside of problem grid (x)"
            assert 0 <= discrete[1] <= self._num_points_axis, "discrete point outside of problem grid (y)"
            x_cont = x_min + discrete[0] * self._grid_resolution
            y_cont = y_min + discrete[1] * self._grid_resolution
            return np.array([x_cont, y_cont])
        else:
            assert discrete.shape[1] == 2, "stacked points must be two-dimensional"
            assert (discrete[:, 0] >= 0).all() and (discrete[:, 0] < self._num_points_axis).all()
            assert (discrete[:, 1] >= 0).all() and (discrete[:, 1] < self._num_points_axis).all()
            x_cont = x_min + discrete[:, 0] * self._grid_resolution
            y_cont = y_min + discrete[:, 1] * self._grid_resolution
            return np.stack((x_cont, y_cont)).T

    def summary(self) -> Dict[str, Any]:
        summary = super(D2TSProblem, self).summary()
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(D2TSProblem, cls).from_summary(json_text)
        return cls(**summary)
