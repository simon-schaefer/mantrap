import time
from typing import List, Tuple

import numpy as np
import pytest
import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.simulation import PotentialFieldStaticSimulation, SocialForcesSimulation
from mantrap.simulation.simulation import GraphBasedSimulation
from mantrap.solver import IGradSolver
from mantrap.utility.io import pytest_is_running
from mantrap.utility.primitives import square_primitives, straight_line_primitive
from mantrap.utility.shaping import check_trajectory_primitives


class PartialIGradSolver(IGradSolver):
    def __init__(self, sim: GraphBasedSimulation, goal: torch.Tensor, objective: str, **solver_params):
        solver_params["verbose"] = not pytest_is_running()
        super(PartialIGradSolver, self).__init__(sim=sim, goal=goal, planning_horizon=5, **solver_params)

        # Get objective and gradient function by name of objective.
        self._obj_function = getattr(IGradSolver, f"_objective_{objective}")
        self._grad_function = getattr(IGradSolver, f"_gradient_{objective}")

    def objective(self, x: np.ndarray) -> float:
        assert self._env.num_ado_modes == 1, "currently only uni-modal agents are supported"
        x2 = torch.from_numpy(x).view(-1, 2)
        assert check_trajectory_primitives(x2, t_horizon=self.T), "x should be ego trajectory"

        objective = self._obj_function(self, x2)

        if self.is_verbose:
            self._x_latest = x.copy()
        return float(objective)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        assert self._env.num_ado_modes == 1, "currently only uni-modal agents are supported"
        x2 = torch.from_numpy(x).view(-1, 2)
        assert check_trajectory_primitives(x2, t_horizon=self.T), "x should be ego trajectory"

        gradient = self._grad_function(self, x2)

        if self.is_verbose:
            self._x_latest = x.copy()  # logging most current optimization values
            self._grad_latest = gradient.copy()
        return gradient

    def constraint_bounds(self, x_init: np.ndarray) -> Tuple[List[float], List[float], List[float], List[float]]:
        lb, ub, cl, cu = super(PartialIGradSolver, self).constraint_bounds(x_init)
        # Optimization variable bounds.
        lb = [-np.inf] * 2 * self.T
        ub = [np.inf] * 2 * self.T
        return lb, ub, cl, cu


###########################################################################
# Tests for iGrad - ipopt optimization ####################################
###########################################################################
@pytest.mark.parametrize(
    "ego_pos, ego_velocity, ado_pos, horizon",
    [
        (torch.tensor([2, 3]), torch.tensor([5, 9]), [torch.tensor([0, 0])], 5),
        (torch.tensor([5, 2]), torch.tensor([1, 1]), [torch.tensor([9, 4]), torch.tensor([3, 2])], 5),
        (torch.tensor([-1, 2]), torch.tensor([0, 0]), [torch.tensor([1, 1])], 7)
    ])
def test_formulation(ego_pos: torch.Tensor, ego_velocity: torch.Tensor, ado_pos: List[torch.Tensor], horizon: int):
    sim = PotentialFieldStaticSimulation(IntegratorDTAgent, {"position": ego_pos, "velocity": ego_velocity})
    for ado_position in ado_pos:
        sim.add_ado(position=ado_position)
    solver = IGradSolver(sim, goal=torch.zeros(2), planning_horizon=horizon)

    # Test interaction objective function
    # Test objective function by comparing a solution trajectory x which is far away and close to the other agents
    # in the scene. Then the close agent is a lot more affected by the ego in the first scenario.
    x = straight_line_primitive(solver.T, sim.ego.position, solver.goal)
    obj_1 = solver._objective_interaction(x)
    x_2 = torch.ones((solver.T, 2)) * 80.0
    obj_2 = solver._objective_interaction(x_2)
    assert obj_1 >= obj_2
    assert np.isclose(obj_2, 0.0)

    # Test output shapes.
    x0 = x.flatten().numpy()
    gradient = solver.gradient(x0)
    jacobian = solver.jacobian(x0)
    # hessian = solver.hessian(x0)
    assert gradient.size == 2 * solver.T
    assert jacobian.size == (solver.T - 1 + 2) * 2 * solver.T
    # assert hessian.shape[0] == hessian.shape[1] == 3 * solver.O
    # assert np.all(np.linalg.eigvals(hessian) >= 0)  # positive semi-definite

    # Test derivatives using derivative-checker from IPOPT framework, format = "mine ~ estimated (difference)".
    if not pytest_is_running():
        solver._solve_optimization(x, approx_jacobian=False, approx_hessian=True, check_derivative=True)


@pytest.mark.parametrize(
    "objective, ego_pos, goal, ado_pos",
    [
        ("interaction", torch.tensor([-5, 0]), torch.tensor([5, 0]), torch.tensor([0, 0.001])),
        ("goal", torch.tensor([2, 2]), torch.tensor([-2, -2]), torch.tensor([0, 1])),
    ])
def test_partial_convergence(objective: str, ego_pos: torch.Tensor, goal: torch.Tensor, ado_pos: torch.Tensor):
    sim = PotentialFieldStaticSimulation(IntegratorDTAgent, {"position": ego_pos}, dt=0.5)
    sim.add_ado(position=ado_pos)
    solver = PartialIGradSolver(sim, goal=goal, objective=objective)

    # Initial trajectory and solver calling.
    x0 = square_primitives(agent=sim.ego, goal=solver.goal, dt=sim.dt)[0, :, :]
    x_optimized = solver._solve_optimization(x0=x0, approx_jacobian=False, max_cpu_time=3.0)
    assert np.isclose(np.linalg.norm(x_optimized[0, :] - x0[0, :]), 0.0, atol=0.1)

    # Objective evaluation.
    x0_obj = solver.objective(x0.flatten().detach().numpy())
    x_opt_obj = solver.objective(x_optimized.flatten().detach().numpy())
    assert x_opt_obj <= x0_obj  # minimization problem (!)


@pytest.mark.parametrize(
    "ego_pos, goal, ado_pos",
    [
        (torch.tensor([-5, 0]), torch.tensor([5, 0]), torch.tensor([0, 0.001])),
        (torch.tensor([0, 0]), torch.tensor([0, 5]), torch.tensor([1, 0])),
        (torch.tensor([2, 2]), torch.tensor([-2, -2]), torch.tensor([0, 1])),
    ])
def test_convergence(ego_pos: torch.Tensor, goal: torch.Tensor, ado_pos: torch.Tensor):
    sim = PotentialFieldStaticSimulation(IntegratorDTAgent, {"position": ego_pos}, dt=0.5)
    sim.add_ado(position=ado_pos)
    solver = IGradSolver(sim, goal=goal, verbose=not pytest_is_running())

    # Initial trajectory and solver calling.
    x0 = square_primitives(agent=sim.ego, goal=solver.goal, dt=sim.dt)[0, :, :]
    x_optimized = solver._solve_optimization(x0=x0, approx_jacobian=False, max_cpu_time=3.0)
    assert np.isclose(np.linalg.norm(x_optimized[0, :] - x0[0, :]), 0.0, atol=0.1)

    # Objective evaluation.
    x0_obj = solver.objective(x0.flatten().detach().numpy())
    x_opt_obj = solver.objective(x_optimized.flatten().detach().numpy())
    assert x_opt_obj <= x0_obj  # minimization problem (!)


def test_gradient_computation_speed():
    sim = SocialForcesSimulation(IntegratorDTAgent, {"position": torch.tensor([-5, 0])}, dt=0.5)
    sim.add_ado(position=torch.tensor([0, 0.001]), goal=torch.tensor([0, -4.0]), num_modes=1)
    solver = PartialIGradSolver(sim, goal=torch.ones(2), objective="interaction")

    # Determine gradient and measure computation time.
    x0 = square_primitives(agent=sim.ego, goal=solver.goal, dt=sim.dt)[0, :, :].flatten().detach().numpy()
    comp_times = []
    for _ in range(10):
        start_time = time.time()
        solver.gradient(x0)
        comp_times.append(time.time() - start_time)
    assert np.mean(comp_times) < 0.07  # faster than 15 Hz (!)


if __name__ == '__main__':
    test_convergence(torch.tensor([-5, 0]), torch.tensor([5, 0]), torch.tensor([0, 0.001]))
