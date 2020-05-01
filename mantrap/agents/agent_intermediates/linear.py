from abc import ABC, abstractmethod
from typing import Tuple

import torch

from mantrap.agents.agent import Agent
from mantrap.utility.maths import Circle


class LinearAgent(Agent, ABC):
    """Intermediate agent class for agents having linear state dynamics.

    For linear state dynamics the dynamics can be computed very efficiently, using matrix-vector multiplication,
    with constant matrices (state-space matrices).

    Passing the simulation time-step directly instead of passing it to every function individually surely is not
    nice, but it enables pre-computing of the agent's dynamics matrices, which speeds up computation of the dynamics
    by at least factor 5 (see examples/tools/timing), especially since the dynamics() method is the most called
    function in the full program. As storing the same information twice generally a bad programming paradigm and
    it reduces the capabilities of the controller a lot to only be able to handle one time-step, as a trade-off
    solution a dictionary of dynamics matrices is built, enabling handling multiple time-steps but still computing them
    once only.

    :param dt: default dynamics time-step [s], default none (no dynamics pre-computation).
    """

    def __init__(self, dt: float = None, **agent_kwargs):
        super(LinearAgent, self).__init__(**agent_kwargs)

        # Passing some time-step `dt` gives the possibility to pre-compute the dynamics matrices for this
        # particular time-step, in order to save computational effort when repeatedly calling the dynamics()
        # method.
        self._dynamics_matrices_dict = {}
        if dt is not None:
            assert dt > 0
            self._dynamics_matrices_dict[dt] = self._dynamics_matrices(dt=dt)

    def _dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float) -> torch.Tensor:
        # Check whether the dynamics matrices have been pre-computed, if not compute them now.
        if dt not in self._dynamics_matrices_dict.keys():
            self._dynamics_matrices_dict[dt] = self._dynamics_matrices(dt=dt)

        A, B, T = self._dynamics_matrices_dict[dt]
        return torch.mv(A, state) + torch.mv(B, action) + T

    @abstractmethod
    def _dynamics_matrices(self, dt: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Determine the state-space/dynamics matrices given integration time-step dt. """
        raise NotImplementedError

    ###########################################################################
    # Reachability ############################################################
    ###########################################################################
    def reachability_boundary(self, time_steps: int, dt: float) -> Circle:
        """Generally for linear agents the N-th can be easily expressed in terms of the controls and the
        initial state by nesting the linear dynamics, resulting in

        .. math:: x_{i + N} = A^N x_i + \sum_{k = 0}^{N - 1} A^{N - k - 1} B u_{i + k}

        Also linear agents are assumed to behave isotropic (here), i.e. the control input can change its direction
        instantly and without limitation. Therefore the forward reachable set within the number of time_steps is
        just a circle (in general ellipse, but agent has same control bounds for both x- and y-direction) around
        the current position, with radius being the maximal allowed agent speed.

        :param time_steps: number of discrete time-steps in reachable time-horizon.
        :param dt: time interval which is assumed to be constant over full path sequence [s].
        """
        if dt not in self._dynamics_matrices_dict.keys():
            self._dynamics_matrices_dict[dt] = self._dynamics_matrices(dt=dt)

        A, B, _ = self._dynamics_matrices_dict[dt]
        x = self.state_with_time
        n = time_steps
        lower, upper = self.control_limits()

        x_n_dict = {}
        x_n_a = torch.mv(A.matrix_power(n), x)
        for name, u in zip(["x_min", "x_max", "y_min", "y_max"],
                           [torch.tensor([lower, 0]), torch.tensor([upper, 0]),
                            torch.tensor([0, lower]), torch.tensor([0, upper])]):
            x_n_b = torch.stack([torch.mv(torch.mm(A.matrix_power(n - k - 1), B), u) for k in range(n)])
            x_n = x_n_a + torch.sum(x_n_b, dim=0)
            x_n_dict[name] = x_n[0] if "x_" in name else x_n[1]

        return Circle.from_min_max_2d(**x_n_dict)
