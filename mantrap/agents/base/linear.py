import abc
import typing

import torch

import mantrap.constants
import mantrap.utility.maths
import mantrap.utility.shaping

from .discrete import DTAgent


class LinearDTAgent(DTAgent, abc.ABC):
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

    def __init__(self, position: torch.Tensor, velocity: torch.Tensor = torch.zeros(2), history: torch.Tensor = None,
                 dt: float = None, max_steps: int = mantrap.constants.AGENT_MAX_PRE_COMPUTATION,  **agent_kwargs
                 ):
        super(LinearDTAgent, self).__init__(position=position, velocity=velocity, history=history, **agent_kwargs)

        # Passing some time-step `dt` gives the possibility to pre-compute the dynamics matrices for this
        # particular time-step, in order to save computational effort when repeatedly calling the dynamics()
        # method.
        self._dynamics_matrices_dict = {}
        self._dynamics_matrices_rolling_dict = {}
        if dt is not None:
            assert dt > 0
            self.dynamics_matrices(dt=dt)
            self.dynamics_rolling_matrices(dt=dt, max_steps=max_steps)

    ###########################################################################
    # Dynamics ################################################################
    ###########################################################################
    def _dynamics(self, state: torch.Tensor, action: torch.Tensor, dt: float) -> torch.Tensor:
        A, B, T = self.dynamics_matrices(dt=dt)
        return torch.mv(A, state) + torch.mv(B, action) + T

    @abc.abstractmethod
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
        assert mantrap.utility.shaping.check_ego_controls(controls)
        assert dt > 0.0
        x_size = self.state_size
        u_size = self.control_size
        t_horizon = controls.shape[0]
        controls = controls.float()

        # Un-squeeze controls if unrolling a single action.
        if len(controls.shape) == 1:
            controls = controls.unsqueeze(dim=0)
        controls_padded = torch.cat((torch.zeros((1, u_size)), controls), dim=0)

        # Determine whole trajectory in single batch computation.
        An, Bn, Tn = self.dynamics_rolling_matrices(dt=dt, max_steps=t_horizon)
        An_horizon = An[:x_size * (t_horizon + 1), :]  # 5 = state-size
        Bn_horizon = Bn[:x_size * (t_horizon + 1), :u_size * (t_horizon + 1)]
        Tn_horizon = Tn[:x_size * (t_horizon + 1)]

        trajectory = torch.mv(An_horizon, self.state_with_time) + \
            torch.mv(Bn_horizon, controls_padded.flatten()) + \
            Tn_horizon * dt
        trajectory = trajectory.view(t_horizon + 1, x_size)

        assert mantrap.utility.shaping.check_ego_trajectory(trajectory, t_horizon + 1, pos_and_vel_only=False)
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
        assert mantrap.utility.shaping.check_ego_trajectory(trajectory, pos_and_vel_only=True)
        assert dt > 0.0

        controls = self._inverse_dynamics_batch(trajectory, dt=dt)

        assert mantrap.utility.shaping.check_ego_controls(controls, t_horizon=trajectory.shape[0] - 1)
        return controls

    ###########################################################################
    # Reachability ############################################################
    ###########################################################################
    def reachability_boundary(self, time_steps: int, dt: float) -> mantrap.utility.maths.Circle:
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
        A, B, _ = self.dynamics_matrices(dt)
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

        return mantrap.utility.maths.Circle.from_min_max_2d(**x_n_dict)

    ###########################################################################
    # Differentiation #########################################################
    ###########################################################################
    def dx_du(self, controls: torch.Tensor, dt: float) -> torch.Tensor:
        """Compute the derivative of the agent's control input with respect to its
        state, evaluated over state trajectory x with discrete state time-step dt.

        As follows from the equations for some state at time-step n `x_n` in dependence of
        the initial state `x0` and the control inputs `u_k`, we have

        .. math::\\frac{dx_i}{du_k} = \\frac{d}{du_k} An * x0 + Bn * uj

        due to the assumptions that the control inputs are independently from each other,
        their derivatives do not depend on each other, i.e. `du_i/du_j = 0 for all i != j`.
        Also the state at time-step n only depends on the initial state and the control
        inputs, not the states in between when the equation above is used !
        Therefore we can simplify the equation above to the following:

        .. math::\\frac{dx_i}{du_k} = \\frac{d}{du_k} Bn_k * u_k = Bn_k

        """
        assert mantrap.utility.shaping.check_ego_controls(controls)
        t_horizon = controls.shape[0]

        _, Bn, _ = self.dynamics_rolling_matrices(dt=dt, max_steps=t_horizon)
        # The trajectory includes the initial state x0, while the matrix Bn assumes to start at time-step
        # t=1, i.e. with x = x1 as first trajectory point. Therefore stack zeros to the beginning of the
        # jacobian matrix.
        Bn0 = torch.zeros((self.state_size, self.control_size * t_horizon))
        Bn1_n = Bn[:self.state_size * t_horizon, :self.control_size * t_horizon]
        return torch.cat((Bn0, Bn1_n))

    ###########################################################################
    # State-Space Representation ##############################################
    ###########################################################################
    def dynamics_rolling_matrices(self, dt: float, max_steps: int
                                  ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Determine matrices for batched trajectory-rolling dynamics using equation shown in the
        definition of the class, stacked to two matrices An and Bn.

        .. math:: An = [I, A, A^2, ..., A^n]
        .. math:: Bn = [[B, 0, ..., 0], [AB, B, 0, ..., 0], ..., [A^{n-1} B, ..., B]]
        .. math:: Tn = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], ..., [0, 0, 0, 0, n]]

        :param dt: dynamics integration time-step [s].
        :param max_steps: maximal number of pre-computed steps.
        """
        def _compute():
            A, B, _ = self._dynamics_matrices(dt=dt)
            A, B = A.float(), B.float()
            x_size = self.state_size
            u_size = self.control_size

            An = torch.cat([A.matrix_power(n) for n in range(0, max_steps + 1)])
            Bn = torch.zeros((x_size * (max_steps + 1), u_size * (max_steps + 1)))
            for m in range(max_steps + 1):
                C = torch.cat([torch.mm(A.matrix_power(k), B) for k in range(max_steps + 1 - m)])
                Bn[x_size * m:, u_size * m:u_size * (m + 1)] = C  # C = m-th column of B

            # Correct for delta time updates (which have been ignored so far).
            Tn = torch.zeros(x_size * (max_steps + 1))
            time_indexes = torch.linspace(1, max_steps + 1, steps=max_steps + 1).long()
            Tn[time_indexes * x_size - 1] = time_indexes.float() - 1  # [0, 1, 2, ...]

            return An.float(), Bn.float(), Tn.float()

        if dt not in self._dynamics_matrices_rolling_dict.keys() or \
            max_steps > self._dynamics_matrices_rolling_dict[dt][2].numel() / 5 - 1:  # 5 = x_size
            self._dynamics_matrices_rolling_dict[dt] = _compute()

        return self._dynamics_matrices_rolling_dict[dt]
