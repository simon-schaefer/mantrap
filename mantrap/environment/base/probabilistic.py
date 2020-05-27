import abc
import typing

import torch

import mantrap.agents
from mantrap.utility.maths import MultiModalDistribution

from .graph_based import GraphBasedEnvironment


class ProbabilisticEnvironment(GraphBasedEnvironment):

    ###########################################################################
    # Simulation graph ########################################################
    ###########################################################################
    def build_connected_graph(self, ego_trajectory: torch.Tensor, return_distribution: bool = False, **kwargs
                              ) -> typing.Tuple[typing.Dict[str, torch.Tensor],
                                                typing.Optional[typing.Dict[str, MultiModalDistribution]]]:
        """Build differentiable graph for predictions over multiple time-steps. For the sake of differentiability
        the computation for the nth time-step cannot be done iteratively, i.e. by determining the current states and
        using the resulting values for computing the next time-step's results in a Markovian manner. Instead the whole
        graph (which is the whole computation) has to be built over n time-steps and evaluated at once by forward pass.

        For building the graph the graphs for each single time-step is built independently while being connected
        using the outputs of the previous time-step and an input for the current time-step. This is quite heavy in
        terms of computational effort and space, however end-to-end-differentiable.

        Build the graph conditioned on some `ego_trajectory`, which is assumed to be fix while the ados in the scene
        behave accordingly, i.e. in reaction to the ego's trajectory. The resulting graph will then contain states
        and controls for every agent in the scene for t in [0, t_horizon], which t_horizon = length of ego trajectory.

        Additionally the `return_distribution` option provides the possibility to get the full distribution
        directly, next to the graph dictionary, which basically builds on parameters of the distribution
        such as the mean, variance, etc.

        :param ego_trajectory: ego's trajectory (t_horizon, 5).
        :param return_distribution: return full distribution as well ?
        :kwargs: additional graph building arguments.
        :return: dictionary over every state of every agent in the scene for t in [0, t_horizon].
        """
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory, pos_and_vel_only=True)
        assert self.ego is not None
        graph, dist = self._build_connected_graph(ego_trajectory, return_distribution=True, **kwargs)
        assert self.check_graph(graph, t_horizon=ego_trajectory.shape[0], include_ego=True)
        return graph if not return_distribution else (graph, dist)

    @abc.abstractmethod
    def _build_connected_graph(self, ego_trajectory: torch.Tensor, **kwargs
                               ) -> typing.Tuple[typing.Dict[str, torch.Tensor],
                                                 typing.Dict[str, MultiModalDistribution]]:
        raise NotImplementedError

    def build_connected_graph_wo_ego(self, t_horizon: int, return_distribution: bool = False, **kwargs
                                     ) -> typing.Tuple[typing.Dict[str, torch.Tensor],
                                                       typing.Optional[typing.Dict[str, MultiModalDistribution]]]:
        """Build differentiable graph for predictions over multiple time-steps. For the sake of differentiability
        the computation for the nth time-step cannot be done iteratively, i.e. by determining the current states and
        using the resulting values for computing the next time-step's results in a Markovian manner. Instead the whole
        graph (which is the whole computation) has to be built over n time-steps and evaluated at once by forward pass.

        For building the graph the graphs for each single time-step is built independently while being connected
        using the outputs of the previous time-step and an input for the current time-step. This is quite heavy in
        terms of computational effort and space, however end-to-end-differentiable.

        Build the graph as if no ego robot would be in the scene, whether or not an ego agent is defined internally.
        Therefore, merely the time-horizon for the predictions (= number of prediction time-steps) is passed.

        Additionally the `return_distribution` option provides the possibility to get the full distribution
        directly, next to the graph dictionary, which basically builds on parameters of the distribution
        such as the mean, variance, etc.

        :param t_horizon: number of prediction time-steps.
        :param return_distribution: return full distribution as well ?
        :kwargs: additional graph building arguments.
        :return: dictionary over every state and control of every ado in the scene for t in [0, t_horizon].
        """
        assert t_horizon > 0
        graph, dist = self._build_connected_graph_wo_ego(t_horizon, return_distribution=True, **kwargs)
        assert self.check_graph(graph, t_horizon=t_horizon, include_ego=False)
        return graph if not return_distribution else (graph, dist)

    @abc.abstractmethod
    def _build_connected_graph_wo_ego(self, t_horizon: int, **kwargs
                                      ) -> typing.Tuple[typing.Dict[str, torch.Tensor],
                                                        typing.Dict[str, MultiModalDistribution]]:
        raise NotImplementedError

    ###########################################################################
    # Simulation parameters ###################################################
    ###########################################################################
    @property
    def is_deterministic(self) -> bool:
        return False
