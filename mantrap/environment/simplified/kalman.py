import typing

import torch
import torch.distributions

import mantrap.constants

from ..base.graph_based import GraphBasedEnvironment


class KalmanEnvironment(GraphBasedEnvironment):
    """Kalman (Filter) - based Environment.

    The Kalman environment implements the update rules, defined in the Kalman Filter, to update the agents
    states iteratively. Thereby no interactions between the agents are taken into account. The growing
    uncertainty in the prediction, growing with the number of predicted time-steps, is modelled as a
    variance increasing with constant rate, which in the Kalman Equations is a constant noise Q_k. Assuming
    constant state space matrices (F, B) we get:

    .. math::x_{k|k-1} = F x_{k-1} + B u_{k-1}

    .. math::P_{k|k-1} = F P_{k-1} F^T + Q

    .. math::K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}

    .. math::x_k = x_{k|k-1} + K_k (z_k - H_k x_{k|k-1})

    .. math::P_k = (I - K_k H_k) P_{k|k-1}

    with z_k = H x_k (H = H_k = eye(N)) the observations, I = eye(n) the identity matrix and R = 0 (perfect
    perception assumption). In the notation used in `mantrap.agents` the state space matrix A is used instead of
    F, but they are equivalent.

    Since no interactions are assumed to ados control inputs (i.e. velocities) are assumed to stay constant over
    the full prediction horizon).
    """

    def _compute_distributions(self, ego_trajectory: typing.Union[typing.List, torch.Tensor],
                               noise_additive: float = mantrap.constants.KALMAN_ADDITIVE_NOISE,
                               **kwargs
                               ) -> typing.Dict[str, torch.distributions.Distribution]:
        """Build a connected graph based on the ego's trajectory.

        The graph should span over the time-horizon of the length of the ego's trajectory and contain the
        positional distribution of every ado in the scene as well as the ego's states itself. When
        possible the graph should be differentiable, such that finding some gradient between the outputted ado
        states and the inputted ego trajectory is determinable.

        Use the Kalman equation to determine the distributions for every time-step in the prediction horizon.
        Note that they are unconditioned on the ego and all the other ados in the scene, as described in the
        class description.

        :param ego_trajectory: ego's trajectory (t_horizon, 5).
        :param noise_additive: additive noise per prediction time-step (Q = diag(noise_additive)).
        :return: ado_id-keyed positional distribution dictionary for times [0, t_horizon].
        """
        t_horizon = len(ego_trajectory) - 1  # works for tensor and list !
        dist_dict = {}

        # Since the agents are not connected with each other anyway in the computation graph, we can not
        # compute any (inter-agent) gradient. Therefore we can simply completely detach the full computation
        # to massively speed up the computation.
        with torch.no_grad():
            for ado in self.ados:
                mus = torch.zeros((t_horizon + 1, 2))
                sigmas = torch.zeros((t_horizon + 1, 2))  # variance will be diagonal for sure (!)

                u_constant = ado.velocity.clone().detach()
                x_k1 = ado.position.clone().detach()  # notation: x_k1 = "x_{k minus 1}"

                # Using the assumptions above (H = eye(), R = 0) the Kalman equations can be vastly simplified,
                # by insertion we get K = eye() and x_k = x_k_k1 for all k.
                F, B, _ = ado.dynamics_matrices(dt=self.dt, x=ado.state)
                F, B = F[0:2, 0:2], B[0:2, :]   # positions only
                FT = torch.transpose(F, 0, 1)
                Q = torch.eye(x_k1.numel()) * noise_additive
                p_k1 = torch.eye(x_k1.numel()) * mantrap.constants.ENV_VAR_INITIAL

                mus[0, :] = x_k1
                sigmas[0, :] = p_k1.diagonal()
                for t in range(t_horizon):
                    x_k = torch.matmul(F, x_k1) + torch.matmul(B, u_constant)
                    p_k = torch.matmul(torch.matmul(F, p_k1), FT) + Q

                    mus[t + 1, :] = x_k
                    sigmas[t + 1, :] = p_k.diagonal()
                    x_k1 = x_k
                    p_k1 = p_k

                dist_dict[ado.id] = torch.distributions.Normal(loc=mus, scale=sigmas)

        return dist_dict

    def _compute_distributions_wo_ego(self, t_horizon: int, **kwargs
                                      ) -> typing.Dict[str, torch.distributions.Distribution]:
        """Build a dictionary of positional distributions for every ado as it would be without the presence
        of a robot in the scene.

        Since no interactions between agents are assumed the conditioned and unconditioned distributions
        are exactly the same.

        :param t_horizon: number of prediction time-steps.
        :kwargs: additional graph building arguments.
        :return: ado_id-keyed positional distribution dictionary for times [0, t_horizon].
        """
        return self._compute_distributions(ego_trajectory=[None] * (t_horizon + 1), **kwargs)

    ###########################################################################
    # Simulation parameters ###################################################
    ###########################################################################
    @property
    def name(self) -> str:
        return "kalman"

    @property
    def is_multi_modal(self) -> bool:
        return False

    @property
    def is_differentiable_wrt_ego(self) -> bool:
        return False
