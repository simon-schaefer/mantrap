import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.simulation import PotentialFieldSimulation
from mantrap.solver import MonteCarloTreeSearch


if __name__ == '__main__':
    sim = PotentialFieldSimulation(IntegratorDTAgent, {"position": torch.tensor([-8, 0])})
    sim.add_ado(position=torch.tensor([0, 0]), velocity=torch.tensor([-1, 0]))
    solver = MonteCarloTreeSearch(sim, goal=torch.tensor([5, 0]))

    solver.determine_ego_controls()
