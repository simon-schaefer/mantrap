from typing import Any, Callable, Dict, Tuple

import numpy as np

from murseco.environment import Environment
from murseco.utility.io import JSONSerializer
from murseco.utility.misc import DictObject


class AbstractProblem(JSONSerializer):
    """The AbstractProblem basically is a template for the most general problem formulation, implementing parameters
    and instances that are used independent of the exact problem formulation (i.e. whether discrete space, time, etc.).
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
        kwargs.update({"name": "problem/abstract/AbstractProblem"})
        super(AbstractProblem, self).__init__(**kwargs)

        assert env.robot is not None, "problem is ill-posed without robot"
        assert env.xaxis[0] <= env.robot.position[0] <= env.xaxis[1], "start position must be inside environment"
        assert env.yaxis[0] <= env.robot.position[1] <= env.yaxis[1], "start position must be inside environment"
        assert env.xaxis[0] <= x_goal[0] <= env.xaxis[1], "goal position must be inside environment"
        assert env.yaxis[0] <= x_goal[1] <= env.yaxis[1], "goal position must be inside environment"
        assert env.robot.state_size == x_goal.size, "initial and end state must have same size"
        assert w_x >= 0, "optimization cost coefficient must be semi-positive"
        assert w_u >= 0, "optimization cost coefficient must be semi-positive"
        assert c_goal >= 0, "final cost must be semi-positive"
        assert risk_max >= 1e-3, "maximal accumulated risk hardly solvable for very small values"
        assert u_max > 0.1, "input norm must be notably positive"
        assert thorizon > 0, "time-horizon for problem must be larger than 0"
        assert dt == env.dt, "time-step should be identical between environment and problem formulation"

        self._env = env
        self._x_goal = x_goal
        self._optim_dict = DictObject(
            {
                "w_x": w_x,
                "w_u": w_u,
                "c_goal": c_goal,
                "risk_max": risk_max,
                "u_max": u_max,
                "dt": env.dt,
                "thorizon": thorizon,
            }
        )

    def generate_trajectory_samples(self, num_samples: int = 10, **generate_kwargs) -> np.ndarray:
        return self._env.generate_trajectory_samples(
            thorizon=self._optim_dict.thorizon, num_samples=num_samples, **generate_kwargs
        )

    @property
    def params(self) -> DictObject:
        return self._optim_dict

    @property
    def environment(self) -> Environment:
        return self._env

    @property
    def robot_dynamics(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        return self._env.robot.dynamics

    @property
    def robot_action_space(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def x_start_goal(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._env.robot.state, self._x_goal

    @property
    def x_size(self) -> int:
        return self._env.robot.state_size

    @property
    def u_size(self) -> int:
        return self._env.robot.input_size

    def summary(self) -> Dict[str, Any]:
        summary = super(AbstractProblem, self).summary()
        env_summary = self._env.summary()
        optim_params_summary = self._optim_dict.__dict__
        summary.update({"env": env_summary, "optim_params": optim_params_summary, "x_goal": self._x_goal.tolist()})
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(AbstractProblem, cls).from_summary(json_text)
        optim_params = json_text["optim_params"]
        x_goal = np.reshape(json_text["x_goal"], (2,))
        summary.update({"env": Environment.from_summary(json_text["env"]), "x_goal": x_goal})
        summary.update(optim_params)
        return summary
