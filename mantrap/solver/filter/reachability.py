import numpy as np
import torch

from mantrap.solver.filter.filter_module import FilterModule


class ReachabilityModule(FilterModule):
    """Filter based on forward reachability analysis between the ego and all ados.

    The reachability deals with the problem that simple euclidean-distance based filtering does not take the agents
    current velocity into account, merely the position. Forward reachability analysis the boundaries of the area
    an agent can reach within some time horizon based on its dynamics and current state (position + velocity).
    When the boundaries of two agents intersect, they could collide within the time-horizon, if they do not there
    is no way they could.

    However forward reachability is fully open-loop, ignoring every kind of interaction between the agents. Using
    backward reachability (closed-loop) would mean to solve the trajectory optimization problem in the filter itself
    and would go beyond the scope of the pre-optimization designed filter.
    """
    def _compute(self) -> np.ndarray:
        with torch.no_grad():
            in_indices = np.zeros(self._env.num_ados)

            ego_boundary = self._env.ego.reachability_boundary(time_steps=self._t_horizon, dt=self._env.dt)
            for m, ado in enumerate(self._env.ados()):
                ado_boundary = ado.reachability_boundary(time_steps=self._t_horizon, dt=self._env.dt)
                in_indices[m] = ego_boundary.does_intersect(ado_boundary)
            in_indices = np.nonzero(in_indices)[0]

        return in_indices
