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
        """Double integrators cannot adapt their velocity instantly, but delayed by (instantly) changing
        their acceleration in any direction. Similarly to the single integrator therefore the forward
        reachable set within the number of time_steps is just a circle (in general ellipse, but agent is
        assumed to be isotropic within this class, i.e. same bounds for both x- and y-direction), just
        not around the current position, since the velocity the agent has acts as an "inertia", shifting
        the center of the circle. The radius of the circle results from the double integrator dynamics,
        the change of position with altering the acceleration to be exact. With `T = time_steps * dt`
        being the time horizon, the reachability bounds are determined for, the circle has the following
        parameters:

        .. math:: center = x(0) + v(0) * T
        .. math:: radius = 0.5 * a_{max} * T^2

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


    # def reachability_boundary(self, time_steps: int, dt: float) -> Circle:
    #     """Single integrators can adapt their velocity instantly in any direction. Therefore the forward
    #     reachable set within the number of time_steps is just a circle (in general ellipse, but agent is
    #     assumed to be isotropic within this class, i.e. same control bounds for both x- and y-direction)
    #     around the current position, with radius being the maximal allowed agent speed.
    #     With `T = time_steps * dt` being the time horizon, the reachability bounds are determined for,
    #     the circle has the following parameters:
    #
    #     .. math:: center = x(0)
    #     .. math:: radius = v_{max} * T
    #
    #     :param time_steps: number of discrete time-steps in reachable time-horizon.
    #     :param dt: time interval which is assumed to be constant over full path sequence [s].
    #     """
    #     return Circle(center=self.position, radius=self.speed_max * dt * time_steps)
