from typing import Dict

import torch

from mantrap.simulation import SocialForcesSimulation


class PotentialFieldStaticSimulation(SocialForcesSimulation):
    """Simplified version of social forces simulation class. The simplified model assumes static agents (ados) in the
    scene, having zero velocity and no incentive to move since goal and position are the same. Hereby, the graph model
    is cut to the pure interaction between ego and ado, no inter-ado interaction and goal pulling force.
    Since the ados would not move at all without an ego agent in the scene, the interaction loss simply comes down
    the the distance of every position of the ados in time to their initial (static) position. """

    def add_ado(self, **ado_kwargs):
        velocity = torch.zeros(2)  # enforce static agent - not moving
        goal = ado_kwargs["position"]  # enforce static agent - no incentive to move
        super(PotentialFieldStaticSimulation, self).add_ado(goal, num_modes=1, velocity=velocity, **ado_kwargs)

    def build_graph(self, ego_state: torch.Tensor, is_intermediate: bool = False, **kwargs) -> Dict[str, torch.Tensor]:
        # Graph initialization - Add ados and ego to graph (position, velocity and goals).
        graph = super(SocialForcesSimulation, self).build_graph(ego_state, is_intermediate, **kwargs)
        k = kwargs["k"] if "k" in kwargs.keys() else 0
        for ghost in self.ado_ghosts:
            graph[f"{ghost.gid}_{k}_goal"] = ghost.goal

        # Make graph with resulting force as an output.
        for ghost in self.ado_ghosts:
            gpos, gvel = graph[f"{ghost.gid}_{k}_position"], graph[f"{ghost.gid}_{k}_velocity"]
            graph[f"{ghost.gid}_{k}_force"] = torch.zeros(2)

            if ego_state is not None:
                ego_pos = graph[f"ego_{k}_position"]
                delta = gpos - ego_pos
                potential_force = torch.sign(delta) * torch.exp(- torch.abs(delta))
                graph[f"{ghost.gid}_{k}_force"] = torch.add(graph[f"{ghost.gid}_{k}_force"], potential_force)

            # Summarize (standard) graph elements.
            graph[f"{ghost.gid}_{k}_output"] = torch.norm(graph[f"{ghost.gid}_{k}_force"])

        return graph
