from abc import abstractmethod
from typing import Dict, Union

import torch

from mantrap.simulation.simulation import Simulation
from mantrap.utility.shaping import check_ego_trajectory, check_trajectories, check_policies, check_weights


class GraphBasedSimulation(Simulation):

    @abstractmethod
    def predict(self, ego_trajectory: torch.Tensor, return_more: bool = False, **graph_kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def predict_wo_ego(self, t_horizon: int, return_more: bool = False, **graph_kwargs) -> torch.Tensor:
        raise NotImplementedError

    ###########################################################################
    # Simulation Graph ########################################################
    ###########################################################################

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

    def transcribe_graph(self, graphs: Dict[str, torch.Tensor], t_horizon: int, returns: bool = False):
        # Remodel simulation outputs, as they are all stored in the simulation graph.
        forces = torch.zeros((self.num_ados, self.num_ado_modes, t_horizon, 2))
        trajectories = torch.zeros((self.num_ados, self.num_ado_modes, t_horizon, 5))
        weights = torch.zeros((self.num_ados, self.num_ado_modes))
        for i in range(self.num_ado_ghosts):
            i_ado, i_mode = self.ghost_to_ado_index(i)
            ghost_id = self.ado_ghosts[i].id
            for k in range(t_horizon):
                trajectories[i_ado, i_mode, k, 0:2] = graphs[f"{ghost_id}_{k}_position"]
                trajectories[i_ado, i_mode, k, 2:4] = graphs[f"{ghost_id}_{k}_velocity"]
                trajectories[i_ado, i_mode, k, -1] = self.sim_time + self.dt * k
                forces[i_ado, i_mode, k, :] = graphs[f"{ghost_id}_{k}_force"]
            weights[i_ado, i_mode] = self.ado_ghosts[i].weight

        assert check_policies(forces, num_ados=self.num_ados, num_modes=self.num_ado_modes, t_horizon=t_horizon)
        assert check_weights(weights, num_ados=self.num_ados, num_modes=self.num_ado_modes)
        assert check_trajectories(trajectories, self.num_ados, t_horizon=t_horizon, modes=self.num_ado_modes)
        return trajectories if not returns else (trajectories, forces, weights)
