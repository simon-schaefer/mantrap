from abc import ABC, abstractmethod
from typing import Tuple

import torch

from mantrap.constants import AGENT_MAX_PRE_COMPUTATION
from mantrap.agents.agent import Agent
from mantrap.utility.maths import Circle
from mantrap.utility.shaping import check_ego_controls, check_ego_trajectory


class LinearAgent(Agent, ABC):
    """Intermediate agent class for agents having linear state dynamics.

    For linear state dynamics the dynamics can be computed very efficiently, using matrix-vector multiplication,
    with constant matrices (state-space matrices).

    .. math:: x_{i + n} = A^n x_i + \\sum_{k=0}^{n-1} A^{n-k-1} * B * u_{i+k}

    Passing the simulation time-step directly instead of passing it to every function individually surely is not
    nice, but it enables pre-computing of the agent's dynamics matrices, which speeds up computation of the dynamics
    by at least factor 5 (see examples/tools/timing), especially since the dynamics() method is the most called
    function in the full program. As storing the same information twice generally a bad programming paradigm and
    it reduces the capabilities of the controller a lot to only be able to handle one time-step, as a trade-off
    solution a dictionary of dynamics matrices is built, enabling handling multiple time-steps but still computing them
    once only.

    :param dt: default dynamics time-step [s], default none (no dynamics pre-computation).
    :param max_steps: maximal number of pre-computed rolling steps.
    """
    def __init__(self, dt: float = None, max_steps: int = AGENT_MAX_PRE_COMPUTATION, **agent_kwargs):
        super(LinearAgent, self).__init__(**agent_kwargs)

        # Passing some time-step `dt` gives the possibility to pre-compute the dynamics matrices for this
        # particular time-step, in order to save computational effort when repeatedly calling the dynamics()
        # method.
        self._dynamics_matrices_dict = {}
        self._dynamics_matrices_rolling_dict = {}
        if dt is not None:
            assert dt > 0
            A, B, T = self._dynamics_matrices(dt=dt)
            self._dynamics_matrices_dict[dt] = (A.float(), B.float(), T.float())
            An, Bn = self._dynamics_rolling_matrices(dt=dt, max_steps=max_steps)
            self._dynamics_matrices_rolling_dict[dt] = (An.float(), Bn.float())

    ###########################################################################
    # Dynamics ################################################################
    ###########################################################################
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

    def _dynamics_rolling_matrices(self, dt: float, max_steps: int):
        """Determine matrices for batched trajectory-rolling dynamics using equation shown in the
        definition of the class, stacked to two matrices An and Bn.

        .. math:: A = [I, A, A^2, ..., A^n]
        .. math:: Bn = [[B, 0, ..., 0], [AB, B, 0, ..., 0], ..., [A^{n-1} B, ..., B]]

        :param max_steps: maximal number of pre-computed steps.
        """
        A, B, _ = self._dynamics_matrices(dt=dt)
        A, B = A.float(), B.float()
        x_size = 5  # state size
        u_size = 2  # control size

        An = torch.cat([A.matrix_power(n) for n in range(0, max_steps + 1)])
        Bn = torch.zeros((x_size * (max_steps + 1), u_size * (max_steps + 1)))
        for n in range(1, max_steps + 1):
            C = torch.cat([torch.mm(A.matrix_power(k), B) for k in range(max_steps + 1 - n)])
            Bn[x_size * n:, u_size * n : u_size * (n + 1)] = C

        return An, Bn

    @abstractmethod
    def _inverse_dynamics_batch(self, batch: torch.Tensor, dt: float) -> torch.Tensor:
        """For linear agents (at least the ones used in this project) the inverse dynamics can
        be batched, i.e. computed with one operation over a whole trajectory.

        :param batch: trajectory-similar state batch (N, 5).
        :returns: controls for the input batch.
        """
        raise NotImplementedError

    ###########################################################################
    # Trajectory ##############################################################
    ###########################################################################
    def unroll_trajectory(self, controls: torch.Tensor, dt: float) -> torch.Tensor:
        """Build the trajectory from some controls and current state, by iteratively applying the model
        dynamics. Thereby a perfect model i.e. without uncertainty and correct is assumed.

        To guarantee that the unrolled trajectory is invertible, i.e. when the resulting trajectory is
        back-transformed to the controls, the same controls should occur. Therefore no checks for the
        feasibility of the controls are made. Also this function is not updating the agent in fact,
        it is rather determining the theoretical trajectory given the agent's dynamics and controls.

        :param controls: sequence of inputs to apply to the robot (N, input_size).
        :param dt: time interval [s] between discrete trajectory states.
        :return: resulting trajectory (no uncertainty in dynamics assumption !), (N, 4).
        """
        assert check_ego_controls(x=controls)
        assert dt > 0.0
        x_size = 5
        u_size = 2
        t_horizon = controls.shape[0]

        # Check whether the dynamics matrices have been pre-computed, if not compute them now.
        if dt not in self._dynamics_matrices_rolling_dict.keys():
            self._dynamics_matrices_rolling_dict[dt] = self._dynamics_rolling_matrices(dt=dt, max_steps=t_horizon)

        # Un-squeeze controls if unrolling a single action.
        if len(controls.shape) == 1:
            controls = controls.unsqueeze(dim=0)
        controls_padded = torch.cat((torch.zeros((1, u_size)), controls), dim=0)

        # Determine whole trajectory in single batch computation.
        An, Bn = self._dynamics_matrices_rolling_dict[dt]
        An_horizon = An[:x_size*(t_horizon + 1), :]  # 5 = state-size
        Bn_horizon = Bn[:x_size*(t_horizon + 1), :u_size*(t_horizon + 1)]  # 2 = control-size

        trajectory = torch.mv(An_horizon, self.state_with_time) + torch.mv(Bn_horizon, controls_padded.flatten())
        trajectory = trajectory.view(t_horizon + 1, x_size)

        assert check_ego_trajectory(trajectory, t_horizon=t_horizon + 1, pos_and_vel_only=False)
        return trajectory

    def roll_trajectory(self, trajectory: torch.Tensor, dt: float) -> torch.Tensor:
        """Determine the controls by iteratively applying the agent's model inverse dynamics.
        Thereby a perfect model i.e. without uncertainty and correct is assumed.

        To guarantee that the unrolled trajectory is invertible, i.e. when the resulting trajectory is
        back-transformed to the controls, the same controls should occur. Therefore no checks for the
        feasibility of the controls are made. Also this function is not updating the agent in fact,
        it is rather determining the theoretical trajectory given the agent's dynamics and controls.

        For linear agents determining the controls from the trajectory is very straight-forward and
        most importantly does not have to be done sequentially (at least not for the types of agents
        used within this project). Therefore the `roll_trajectory()` method can be improved.

        :param trajectory: sequence of states to apply to the robot (N, 4).
        :param dt: time interval [s] between discrete trajectory states.
        :return: inferred controls (no uncertainty in dynamics assumption !), (N, input_size).
        """
        assert check_ego_trajectory(trajectory, pos_and_vel_only=True)
        assert dt > 0.0

        controls = self._inverse_dynamics_batch(trajectory, dt=dt)

        assert check_ego_controls(controls, t_horizon=trajectory.shape[0] - 1)
        return controls

    ###########################################################################
    # Reachability ############################################################
    ###########################################################################
    def reachability_boundary(self, time_steps: int, dt: float) -> Circle:
        """Generally for linear agents the N-th can be easily expressed in terms of the controls and the
        initial state by nesting the linear dynamics, resulting in

        .. math:: x_{i + N} = A^N x_i + \\sum_{k = 0}^{N - 1} A^{N - k - 1} B u_{i + k}

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
        A, B = A.float(), B.float()
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
