import numpy as np
import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.simulation import PotentialFieldStaticSimulation
from mantrap.solver import IGradSolver


def test_formulation():
    sim = PotentialFieldStaticSimulation(IntegratorDTAgent, {"position": torch.tensor([-5, 0])})
    sim.add_ado(position=torch.tensor([0, 0]))
    solver = IGradSolver(sim, goal=torch.tensor([5, 0]))

    # Test objective function.
    # Test objective function by comparing a solution trajectory x which is far away and close to the other agents
    # in the scene. Then the close agent is a lot more affected by the ego in the first scenario.
    obj_1 = solver.objective(x=np.array([0, 0.1]))
    obj_2 = solver.objective(x=np.array([0, 8.0]))
    assert obj_1 >= obj_2
    assert np.isclose(obj_2, 0.0, atol=1.0)

    # Test gradient function.
    grad = solver.gradient(x=np.array([0, 0.1]))


if __name__ == '__main__':
    test_formulation()
