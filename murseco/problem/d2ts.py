from typing import Any, Dict, List, Tuple

import cvxpy as cp
import numpy as np

from murseco.environment import Environment
from murseco.utility.io import JSONSerializer
from murseco.utility.misc import DictObject


class D2TSProblem(JSONSerializer):
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
        mproc: bool = True,
        grid_resolution: float = 0.01,
        **kwargs,
    ):
        kwargs.update({"name": "problem/d2ts/D2TSProblem"})
        super(D2TSProblem, self).__init__(**kwargs)

        assert env.robot is not None, "problem is ill-posed without robot"
        assert env.xaxis[0] <= env.robot.position[0] <= env.xaxis[1], "start position must be inside environment"
        assert env.yaxis[0] <= env.robot.position[1] <= env.yaxis[1], "start position must be inside environment"
        assert env.xaxis[0] <= x_goal[0] <= env.xaxis[1], "goal position must be inside environment"
        assert env.yaxis[0] <= x_goal[1] <= env.yaxis[1], "goal position must be inside environment"
        assert w_x >= 0, "optimization cost coefficient must be semi-positive"
        assert w_u >= 0, "optimization cost coefficient must be semi-positive"
        assert c_goal >= 0, "final cost must be semi-positive"
        assert risk_max >= 1e-3, "maximal accumulated risk hardly solvable for very small values"
        assert u_max > 0.1, "input norm must be notably positive"
        assert thorizon > 0, "time-horizon for problem must be larger than 0"

        self._env = env
        self._x_goal = x_goal
        self._optim_dict = DictObject(
            {"w_x": w_x, "w_u": w_u, "c_goal": c_goal, "risk_max": risk_max, "u_max": u_max, "thorizon": thorizon}
        )

        # Temporal and spatial discretization.
        assert env.xaxis == env.yaxis, "discretization assumes equal axes for simplification (and nicer graphs)"
        num_points_per_axis = int((env.xaxis[1] - env.xaxis[0]) / grid_resolution)
        self._tppdf, self._meshgrid = env.tppdf(num_points=num_points_per_axis, mproc=mproc)

    @property
    def params(self):
        return self._optim_dict

    @property
    def environment(self) -> Environment:
        return self._env

    @property
    def x_start_goal(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._env.robot.state, self._x_goal

    @property
    def grid(self) -> Tuple[List[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        return self._tppdf, self._meshgrid

    def summary(self) -> Dict[str, Any]:
        summary = super(D2TSProblem, self).summary()
        env_summary = self._env.summary()
        optim_params_summary = self._optim_dict.__dict__
        summary.update({"env": env_summary, "optim_params": optim_params_summary, "x_goal": self._x_goal.tolist()})
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(D2TSProblem, cls).from_summary(json_text)
        optim_params = json_text["optim_params"]
        x_goal = np.reshape(json_text["x_goal"], (2,))
        summary.update({"env": Environment.from_summary(json_text["env"]), "x_goal": x_goal})
        summary.update(optim_params)
        return cls(**summary)
