import time

import numpy as np
import pytest
import torch

from mantrap.constants import constraint_min_distance
from mantrap.agents import IntegratorDTAgent
from mantrap.simulation import PotentialFieldSimulation
from mantrap.simulation.graph_based import GraphBasedSimulation
from mantrap.solver import CGradSolver, IGradSolver, SGradSolver
from mantrap.solver.ipopt_solver import IPOPTSolver


@pytest.mark.parametrize(
    "solver_class, test_kwargs",
    [
        (IGradSolver, {"T": 5, "num_constraints": 10, "num_control_points": 1}),
        (IGradSolver, {"T": 10, "num_constraints": 20, "num_control_points": 2}),
        (IGradSolver, {"T": 10, "num_constraints": 20, "num_control_points": 4}),
        (SGradSolver, {"T": 10, "num_constraints": 20}),
        (SGradSolver, {"T": 5, "num_constraints": 10}),
        (CGradSolver, {"T": 10, "num_constraints": 20 + 2 * 10})
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
        z0 = solver.z0s_default().detach().numpy()

        # Test gradient function.
        grad = solver.gradient(z=z0)
        assert np.linalg.norm(grad) > 0

        # Test output shapes.
        constraints = solver.constraints(z0)
        gradient = solver.gradient(z0)
        jacobian = solver.jacobian(z0)
        assert constraints.size == test_kwargs["num_constraints"]
        assert gradient.size == z0.flatten().size
        assert jacobian.size == test_kwargs["num_constraints"] * z0.flatten().size

        # Test derivatives using derivative-checker from IPOPT framework, format = "mine ~ estimated (difference)".
        if solver.is_verbose:
            x0 = torch.from_numpy(z0)
            solver.solve_single_optimization(x0, approx_jacobian=False, approx_hessian=True, check_derivative=True)

    @staticmethod
    def test_single_agent_scenario(solver_class: IPOPTSolver.__class__, test_kwargs):
        sim = PotentialFieldSimulation(IntegratorDTAgent, {"position": torch.tensor([-8, 0])})
        sim.add_ado(position=torch.tensor([0, 0]), velocity=torch.tensor([-1, 0]))
        solver = solver_class(sim, goal=torch.tensor([8, 0]), **test_kwargs)

        x4_opt = solver.solve_single_optimization(z0=solver.z0s_default(), approx_jacobian=False, approx_hessian=True)
        TestIPOPTSolvers.check_output_trajectory(x4_opt, sim=sim, solver=solver)

    @staticmethod
    def test_multiple_agent_scenario(solver_class: IPOPTSolver.__class__, test_kwargs):
        sim = PotentialFieldSimulation(IntegratorDTAgent, {"position": torch.tensor([-8, 0])})
        sim.add_ado(position=torch.tensor([0, 0]), velocity=torch.tensor([-1, 0]))
        sim.add_ado(position=torch.tensor([3, 2]), velocity=torch.tensor([0.1, -1.5]))
        sim.add_ado(position=torch.tensor([3, -8]), velocity=torch.tensor([2.5, 1.5]))
        solver = solver_class(sim, goal=torch.tensor([8, 0]), **test_kwargs)

        x_opt = solver.solve_single_optimization(z0=solver.z0s_default(), approx_jacobian=False, approx_hessian=True)
        TestIPOPTSolvers.check_output_trajectory(x_opt, sim=sim, solver=solver)

    @staticmethod
    def test_x_to_x2(solver_class: IPOPTSolver.__class__, test_kwargs):
        sim = PotentialFieldSimulation(IntegratorDTAgent, {"position": torch.tensor([-5, 0])})
        sim.add_ado(position=torch.zeros(2))
        solver = solver_class(sim, goal=torch.tensor([5, 0]), **test_kwargs)
        z0 = solver.z0s_default().detach().numpy().flatten()
        x0 = solver.z_to_ego_trajectory(z0).detach().numpy()[:, 0:2]

        x02 = np.reshape(x0, (-1, 2))
        for xi in x02:
            assert any([np.isclose(np.linalg.norm(xk - xi), 0, atol=2.0) for xk in x0])

    @staticmethod
    def test_computation_speed(solver_class: IPOPTSolver.__class__, test_kwargs):
        sim = PotentialFieldSimulation(IntegratorDTAgent, {"position": torch.tensor([-5, 0])})
        sim.add_ado(position=torch.tensor([0, 0]), velocity=torch.tensor([-1, 0]))
        solver = solver_class(sim, goal=torch.tensor([5, 0]), **test_kwargs)
        x = solver.z0s_default().detach().numpy()

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
        assert torch.norm(x[0, 0:2] - sim.ego.position).item() < 1e-3
        assert torch.norm(x[-1, 0:2] - solver.goal) <= torch.norm(x[0, 0:2] - solver.goal)

        if solver.constraints_fulfilled():  # optimization variable bound constraint
            assert torch.all(sim.axes[0][0] <= x[:, 0]) and torch.all(x[:, 0] <= sim.axes[0][1])
            assert torch.all(sim.axes[1][0] <= x[:, 1]) and torch.all(x[:, 1] <= sim.axes[1][1])


def test_c_grad_solver():
    env = PotentialFieldSimulation(IntegratorDTAgent, {"position": torch.tensor([-5, 0.5])})
    env.add_ado(position=torch.tensor([0, 0]), velocity=torch.tensor([-1, 0]))
    c_grad_solver = CGradSolver(env, goal=torch.tensor([5, 0]), T=10, verbose=True)

    from mantrap.utility.primitives import square_primitives
    x0 = square_primitives(start=env.ego.position, end=c_grad_solver.goal, dt=env.dt, steps=10)[1, :, :]
    x40 = env.ego.expand_trajectory(x0, dt=env.dt)
    u0 = env.ego.roll_trajectory(x40, dt=env.dt)

    x_solution = c_grad_solver.solve_single_optimization(z0=u0, max_cpu_time=10.0)
    ado_trajectories = env.predict_w_trajectory(trajectory=x_solution)

    for t in range(10):
        for m in range(env.num_ado_ghosts):
            assert torch.norm(x_solution[t, 0:2] - ado_trajectories[m, :, t, 0:2]).item() >= constraint_min_distance
