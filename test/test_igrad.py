from typing import List

import numpy as np
import pytest
import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.simulation import PotentialFieldSimulation
from mantrap.solver import IGradSolver
from mantrap.utility.io import no_pytest


@pytest.mark.parametrize(
    "ego_pos, ego_goal, ado_pos, horizon",
    [
        (torch.tensor([-5, 0]), torch.tensor([5, 0]), [torch.tensor([0, 0])], 5),
        (torch.tensor([2, 3]), torch.tensor([5, 9]), [torch.tensor([0, 0])], 5),
        (torch.tensor([-5, 2]), torch.tensor([1, 1]), [torch.tensor([9, 4]), torch.tensor([3, 2])], 5),
    ],
)
def test_formulation(ego_pos: torch.Tensor, ego_goal: torch.Tensor, ado_pos: List[torch.Tensor], horizon: int):
    sim = PotentialFieldSimulation(IntegratorDTAgent, {"position": ego_pos})
    for pos in ado_pos:
        sim.add_ado(position=pos)
    solver = IGradSolver(sim, goal=ego_goal, prediction_horizon=horizon, verbose=no_pytest(), num_control_points=1)

    # Test objective function.
    # Test objective function by comparing a solution trajectory x which is far away and close to the other agents
    # in the scene. Then the close agent is a lot more affected by the ego in the first scenario.
    obj_1 = solver.objective(x=np.array([2.43, 0.89]))
    obj_2 = solver.objective(x=np.array([10.0, 10.0]))
    assert obj_1 >= obj_2

    # Test gradient function.
    grad = solver.gradient(x=np.array([0.4, 0.1]))
    assert np.linalg.norm(grad) > 0

    # Test output shapes.
    x0 = np.array([0.4, 2])
    constraints = solver.constraints(x0)
    gradient = solver.gradient(x0)
    jacobian = solver.jacobian(x0)
    assert constraints.size == (solver.T - 1)
    assert gradient.size == 2
    assert jacobian.size == (solver.T - 1) * 2 * solver.num_control_points

    # Test derivatives using derivative-checker from IPOPT framework, format = "mine ~ estimated (difference)".
    if no_pytest():
        x0 = torch.tensor([0.4, 2])
        solver._solve_optimization(x0, approx_jacobian=False, approx_hessian=True, check_derivative=True)


if __name__ == '__main__':
    test_formulation(torch.tensor([-1, 2]), torch.tensor([0, 0]), [torch.tensor([1, 1])], 7)


def test_single_agent_scenario():
    sim = PotentialFieldSimulation(IntegratorDTAgent, {"position": torch.tensor([-8, 0])})
    sim.add_ado(position=torch.tensor([0, 0]), velocity=torch.tensor([-1, 0]))
    solver = IGradSolver(sim, goal=torch.tensor([8, 0]), verbose=no_pytest(), planning_horizon=4, num_control_points=1)

    x = solver._solve_optimization(x0=torch.tensor([0.1, 0.1]), approx_jacobian=False, approx_hessian=True)

    assert torch.norm(x[0, :] - sim.ego.position).item() < 1e-3
    assert torch.norm(x[-1, :] - solver.goal).item() < 1e-3
    assert torch.all(sim.axes[0][0] <= x[:, 0]) and torch.all(x[:, 0] <= sim.axes[0][1])
    assert torch.all(sim.axes[1][0] <= x[:, 1]) and torch.all(x[:, 1] <= sim.axes[1][1])


def test_multiple_agent_scenario():
    sim = PotentialFieldSimulation(IntegratorDTAgent, {"position": torch.tensor([-8, 0])})
    sim.add_ado(position=torch.tensor([0, 0]), velocity=torch.tensor([-1, 0]))
    sim.add_ado(position=torch.tensor([3, 2]), velocity=torch.tensor([0.1, -1.5]))
    sim.add_ado(position=torch.tensor([3, -8]), velocity=torch.tensor([2.5, 1.5]))
    solver = IGradSolver(sim, torch.tensor([8, 0]), verbose=no_pytest(), planning_horizon=10)

    x0 = torch.stack((torch.tensor([0.1, -0.01]), torch.tensor([4.0, 0.0])))
    x = solver._solve_optimization(x0=x0, approx_jacobian=False, approx_hessian=True)

    assert torch.norm(x[0, :] - sim.ego.position).item() < 1e-3
    assert torch.norm(x[-1, :] - solver.goal).item() < 1e-3
    assert torch.all(sim.axes[0][0] <= x[:, 0]) and torch.all(x[:, 0] <= sim.axes[0][1])
    assert torch.all(sim.axes[1][0] <= x[:, 1]) and torch.all(x[:, 1] <= sim.axes[1][1])


def test_x_to_x2():
    sim = PotentialFieldSimulation(IntegratorDTAgent, {"position": torch.tensor([-5, 0])})
    solver = IGradSolver(sim, goal=torch.tensor([5, 0]), planning_horizon=10, num_control_points=1)

    x0 = np.array([3, 5])
    x = solver.x_to_ego_trajectory(x0).detach().numpy()

    assert any([np.isclose(np.linalg.norm(xk - x0), 0, atol=1.0) for xk in x])
    assert any([np.isclose(np.linalg.norm(xk - sim.ego.position.detach().numpy()), 0, atol=0.1) for xk in x])
    assert any([np.isclose(np.linalg.norm(xk - solver.goal.detach().numpy()), 0, atol=0.1) for xk in x])
