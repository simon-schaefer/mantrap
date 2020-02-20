from abc import abstractmethod
from typing import Dict, Tuple, Union

import torch

from mantrap.simulation.simulation import Simulation
from mantrap.utility.shaping import check_ego_trajectory


class GraphBasedSimulation(Simulation):

    @abstractmethod
    def predict(
        self, graph_input: Union[int, torch.Tensor], returns: bool = False, **graph_kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def build_graph(self, ego_state: torch.Tensor = None, **graph_kwargs) -> Dict[str, torch.Tensor]:
        k = graph_kwargs["k"] if "k" in graph_kwargs.keys() else 0
        ado_grad = graph_kwargs["ado_grad"] if "ado_grad" in graph_kwargs.keys() else False
        ego_grad = graph_kwargs["ego_grad"] if "ego_grad" in graph_kwargs.keys() else True
        graph = {}

        for ghost in self.ado_ghosts:
            graph[f"{ghost.id}_{k}_position"] = ghost.agent.position
            graph[f"{ghost.id}_{k}_velocity"] = ghost.agent.velocity
            if ado_grad and graph[f"{ghost.id}_{k}_position"].requires_grad is not True:
                graph[f"{ghost.id}_{k}_position"].requires_grad = True
                graph[f"{ghost.id}_{k}_velocity"].requires_grad = True

        if ego_state is not None:
            graph[f"ego_{k}_position"] = ego_state[0:2]
            graph[f"ego_{k}_velocity"] = ego_state[2:4]

            if ego_grad:
                graph[f"ego_{k}_position"].requires_grad = True
                graph[f"ego_{k}_velocity"].requires_grad = True

        return graph

    @abstractmethod
    def build_connected_graph(self, graph_input: Union[int, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Build differentiable graph for predictions over multiple time-steps. For the sake of differentiability
         the computation for the nth time-step cannot be done iteratively, i.e. by determining the current states and
         using the resulting values for computing the next time-step's results in a Markovian manner. Instead the whole
         graph (which is the whole computation) has to be built over n time-steps and evaluated at once by forward pass.

         For building the graph the graphs for each single time-step is built independently while being connected
         using the outputs of the previous time-step and an input for the current time-step. This is quite heavy in
         terms of computational effort and space, however end-to-end-differentiable.

        :param graph_input: either ego path (positions) or t_horizon.
        """
        if type(graph_input) == torch.Tensor:  # ego_path
            assert check_ego_trajectory(graph_input, pos_only=True)
            assert self.ego is not None, "ego must be defined to get a path allocated"
        else:  # t_horizon
            assert graph_input > 0, "invalid prediction time horizon, must be larger zero"
        return dict()
