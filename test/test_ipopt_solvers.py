import time

import numpy as np
import pytest
import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.simulation import PotentialFieldSimulation
from mantrap.simulation.simulation import GraphBasedSimulation
from mantrap.solver import IGradSolver, SGradSolver
from mantrap.solver.ipopt_solver import IPOPTSolver


@pytest.mark.parametrize(
    "solver_class, test_kwargs",
    [
        (IGradSolver, {"T": 5, "num_constraints": 4, "num_control_points": 1}),
        (IGradSolver, {"T": 10, "num_constraints": 9, "num_control_points": 2}),
        (IGradSolver, {"T": 10, "num_constraints": 9, "num_control_points": 4}),
        (SGradSolver, {"T": 10, "num_constraints": 11}),
        (SGradSolver, {"T": 5, "num_constraints": 6}),
    ]
)
class TestIPOPTSolvers:

    @staticmethod
    def test_formulation(solver_class: IPOPTSolver.__class__, test_kwargs):
        ego_pos = torch.tensor([-5, 2])
        ego_goal = torch.tensor([1, 1])
        ado_poses = [torch.tensor([9, 4]), torch.tensor([3, 2])]

        sim = PotentialFieldSimulation(IntegratorDTAgent, {"position": ego_pos})
        for pos in ado_poses:
            sim.add_ado(position=pos)
        solver = solver_class(sim, goal=ego_goal, **test_kwargs)
        x = solver.x0_default().detach().numpy()

        # Test gradient function.
        grad = solver.gradient(x=x)
        assert np.linalg.norm(grad) > 0

        # Test output shapes.
        constraints = solver.constraints(x)
        gradient = solver.gradient(x)
        jacobian = solver.jacobian(x)
        assert constraints.size == test_kwargs["num_constraints"]
        assert gradient.size == x.flatten().size
        assert jacobian.size == test_kwargs["num_constraints"] * x.flatten().size

        # Test derivatives using derivative-checker from IPOPT framework, format = "mine ~ estimated (difference)".
        if solver.is_verbose:
            x0 = torch.from_numpy(x)
            solver._solve_optimization(x0, approx_jacobian=False, approx_hessian=True, check_derivative=True)

    @staticmethod
    def test_single_agent_scenario(solver_class: IPOPTSolver.__class__, test_kwargs):
        sim = PotentialFieldSimulation(IntegratorDTAgent, {"position": torch.tensor([-8, 0])})
        sim.add_ado(position=torch.tensor([0, 0]), velocity=torch.tensor([-1, 0]))
        solver = solver_class(sim, goal=torch.tensor([8, 0]), **test_kwargs)

        x_opt = solver._solve_optimization(x0=solver.x0_default(), approx_jacobian=False, approx_hessian=True)
        TestIPOPTSolvers.check_output_trajectory(x_opt, sim=sim, solver=solver)

    @staticmethod
    def test_multiple_agent_scenario(solver_class: IPOPTSolver.__class__, test_kwargs):
        sim = PotentialFieldSimulation(IntegratorDTAgent, {"position": torch.tensor([-8, 0])})
        sim.add_ado(position=torch.tensor([0, 0]), velocity=torch.tensor([-1, 0]))
        sim.add_ado(position=torch.tensor([3, 2]), velocity=torch.tensor([0.1, -1.5]))
        sim.add_ado(position=torch.tensor([3, -8]), velocity=torch.tensor([2.5, 1.5]))
        solver = solver_class(sim, goal=torch.tensor([8, 0]), **test_kwargs)

        x_opt = solver._solve_optimization(x0=solver.x0_default(), approx_jacobian=False, approx_hessian=True)
        TestIPOPTSolvers.check_output_trajectory(x_opt, sim=sim, solver=solver)

    @staticmethod
    def test_x_to_x2(solver_class: IPOPTSolver.__class__, test_kwargs):
        sim = PotentialFieldSimulation(IntegratorDTAgent, {"position": torch.tensor([-5, 0])})
        solver = solver_class(sim, goal=torch.tensor([5, 0]), **test_kwargs)
        x = solver.x0_default().detach().numpy().flatten()
        x_opt = solver.x_to_ego_trajectory(x).detach().numpy()

        x_in = np.reshape(x, (-1, 2))
        for xi in x_in:
            assert any([np.isclose(np.linalg.norm(xk - xi), 0, atol=2.0) for xk in x_opt])

    @staticmethod
    def test_computation_speed(solver_class: IPOPTSolver.__class__, test_kwargs):
        sim = PotentialFieldSimulation(IntegratorDTAgent, {"position": torch.tensor([-5, 0])})
        sim.add_ado(position=torch.tensor([0, 0]), velocity=torch.tensor([-1, 0]))
        solver = solver_class(sim, goal=torch.tensor([5, 0]), **test_kwargs)
        x = solver.x0_default().detach().numpy()

        # Determine objective/gradient and measure computation time.
        comp_times_objective = []
        comp_times_gradient = []

        for _ in range(10):
            start_time = time.time()
            solver.gradient(x)
            comp_times_gradient.append(time.time() - start_time)
            start_time = time.time()
            solver.objective(x)
            comp_times_objective.append(time.time() - start_time)
        assert np.mean(comp_times_objective) < 0.04  # faster than 25 Hz (!)
        assert np.mean(comp_times_gradient) < 0.07  # faster than 15 Hz (!)

    @staticmethod
    def check_output_trajectory(x: torch.Tensor, sim: GraphBasedSimulation, solver: IPOPTSolver):
        assert torch.norm(x[0, :] - sim.ego.position).item() < 1e-3
        assert torch.norm(x[-1, :] - solver.goal) <= torch.norm(x[0, :] - solver.goal)

        if solver.constraints_fulfilled():  # optimization variable bound constraint
            assert torch.all(sim.axes[0][0] <= x[:, 0]) and torch.all(x[:, 0] <= sim.axes[0][1])
            assert torch.all(sim.axes[1][0] <= x[:, 1]) and torch.all(x[:, 1] <= sim.axes[1][1])
