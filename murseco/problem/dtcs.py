from typing import Any, Dict, List

import numpy as np

from murseco.environment import Environment
from murseco.problem.abstract import AbstractProblem
from murseco.utility.stats import GMM2D


class DTCSProblem(AbstractProblem):
    """The CTDSProblem (CTDS) Problem defines the optimization problem as following, for the time-step
    0 <= k <= N, the continuous space vector x_k, the system dynamics f(x_k, u_k) and the time and spatially dependent
    continuous probability density function of the risk pdf(x_k, k):

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
        **kwargs,
    ):
        kwargs.update({"name": "problem/dtcs/DTCSProblem"})
        super(DTCSProblem, self).__init__(
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

        # Position-based obstacle probability distribution function.
        self._tppdf = env.tppdf(thorizon=thorizon, mproc=mproc)

    @property
    def tppdf(self) -> List[GMM2D]:
        return self._tppdf

    def summary(self) -> Dict[str, Any]:
        summary = super(DTCSProblem, self).summary()
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(DTCSProblem, cls).from_summary(json_text)
        return cls(**summary)
