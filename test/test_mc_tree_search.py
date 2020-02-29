import torch

from mantrap.agents import DoubleIntegratorDTAgent
from mantrap.simulation import PotentialFieldSimulation
from mantrap.solver import MonteCarloTreeSearch
from mantrap.utility.shaping import check_ego_controls


def test_single_agent_scenario():
    sim = PotentialFieldSimulation(DoubleIntegratorDTAgent, {"position": torch.tensor([-5, 0])})
    sim.add_ado(position=torch.tensor([0, 0]), velocity=torch.tensor([-1, 0]))
    solver = MonteCarloTreeSearch(sim, goal=torch.tensor([5, 0]), t_planning=10, verbose=True)
    controls = solver.determine_ego_controls()

    assert check_ego_controls(controls, t_horizon=9)
