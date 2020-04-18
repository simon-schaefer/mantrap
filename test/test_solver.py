import time

import numpy as np
import pytest
import torch

from mantrap.constants import *
from mantrap.agents import IntegratorDTAgent
from mantrap.environment import ENVIRONMENTS
from mantrap.environment.environment import GraphBasedEnvironment
from mantrap.solver import *
from mantrap.solver.solver import Solver
from mantrap.solver.ipopt_solver import IPOPTSolver
from mantrap.utility.shaping import check_ego_trajectory, check_ado_trajectories


def scenario(solver_class: Solver.__class__, env_class: GraphBasedEnvironment.__class__, **solver_kwargs):
    env = env_class(IntegratorDTAgent, {"position": torch.tensor([-5, 2])})
    env.add_ado(position=torch.tensor([3, 2]))
    solver = solver_class(env, goal=torch.tensor([1, 1]), **solver_kwargs)
    z0 = solver.z0s_default(just_one=True).detach().numpy()
    return env, solver, z0

# def test_eval_environment():
#     env = PotentialFieldEnvironment(IntegratorDTAgent, {"position": torch.tensor([-8, 0])})
#     env.add_ado(position=torch.tensor([0, 0]), velocity=torch.tensor([-1, 0.2]))
#     env.add_ado(position=torch.ones(2), velocity=torch.tensor([-5, 4.2]))
#
#     # Since the determine ego controls are predictable in the constant solver, the output ego trajectory basically
#     # is the unrolled trajectory by applying the same controls again and again.
#     time_steps = 5
#     ego_controls = torch.tensor([1, 0] * time_steps).view(-1, 2)
#     ego_trajectory_opt_exp = env.ego.unroll_trajectory(controls=ego_controls, dt=env.dt)
#     ado_trajectories_exp_sim = env.predict_w_controls(ego_controls=ego_controls)
#
#     # First dont pass evaluation environment, then it planning and evaluation environment should be equal.
#     # Ado Trajectories -> just the actual positions (first entry of planning trajectory at every time-step)
#     solver = ConstantSolver(env, goal=torch.zeros(2), t_planning=2, objectives=[], constraints=[])
#     ego_trajectory_opt, ado_trajectories = solver.solve(time_steps=time_steps)
#     assert torch.all(torch.isclose(ego_trajectory_opt, ego_trajectory_opt_exp))
#     assert torch.all(torch.isclose(ado_trajectories, ado_trajectories_exp_sim, atol=0.1))
#
#     # Second pass evaluation environment. Since the environment updates should be performed using the evaluation
#     # environment the ado trajectories should be equal to predicting the ado behaviour given the (known) ego controls.
#     # To test multi-modality but still have a deterministic mode collapse all weight is put on one mode.
#     # However, since the agents themselves are the same, especially the ego agent, the ego trajectory should be the
#     # same as in the first test ==> (only one mode since deterministic)
#     eval_env = SocialForcesEnvironment(IntegratorDTAgent, {"position": torch.tensor([-8, 0])})
#     mode_kwargs = {"num_modes": 2, "weights": [1.0, 0.0]}
#     eval_env.add_ado(position=torch.zeros(2), velocity=torch.tensor([-1, 0.2]), goal=torch.ones(2) * 10, **mode_kwargs)
#     eval_env.add_ado(position=torch.ones(2), velocity=torch.tensor([-5, 4.2]), goal=torch.ones(2) * 10, **mode_kwargs)
#     ado_trajectories_exp_eval = eval_env.predict_w_controls(ego_controls=ego_controls)[:, :1, :, :]  # first mode only
#     solver = ConstantSolver(env, eval_env=eval_env, goal=torch.zeros(2), t_planning=2, objectives=[], constraints=[])
#     ego_trajectory_opt, ado_trajectories = solver.solve(time_steps=time_steps)
#     assert torch.all(torch.isclose(ego_trajectory_opt, ego_trajectory_opt_exp))
#     assert torch.all(torch.isclose(ado_trajectories, ado_trajectories_exp_eval, atol=0.1))


###########################################################################
# Tests - All Solvers #####################################################
###########################################################################
@pytest.mark.parametrize("solver_class", SOLVER)
@pytest.mark.parametrize("env_class", ENVIRONMENTS)
class TestSolvers:

    @staticmethod
    def test_convergence(solver_class: Solver.__class__, env_class: GraphBasedEnvironment.__class__):
        ego_goal_distance = (AGENT_SPEED_MAX / 2) * ENV_DT_DEFAULT
        env = env_class(IntegratorDTAgent, {"position": torch.tensor([-ego_goal_distance, 0])}, dt=ENV_DT_DEFAULT)
        solver = solver_class(env, goal=torch.zeros(2), t_planning=1, objectives=[("goal", 1.0)], constraints=[])

        z0 = solver.z0s_default(just_one=True)
        z_opt, _, _ = solver.optimize(z0=z0, tag="core0", max_cpu_time=1.0, max_iter=1000)
        ego_controls = solver.z_to_ego_controls(z=z_opt.detach().numpy())
        ego_trajectory_opt = solver.env.ego.unroll_trajectory(controls=ego_controls, dt=solver.env.dt)

        assert torch.all(torch.isclose(ego_trajectory_opt[0, :], env.ego.state_with_time))
        for k in range(1, solver.T):
            assert torch.all(torch.isclose(ego_trajectory_opt[k, 0:2], solver.goal, atol=0.5))

    @staticmethod
    def test_formulation(solver_class: Solver.__class__, env_class: GraphBasedEnvironment.__class__):
        env, solver, z0 = scenario(solver_class, env_class=env_class)

        # Test output shapes.
        objective = solver.objective(z=z0, tag="core0")
        assert type(objective) == float
        constraints = solver.constraints(z=z0, tag="core0", return_violation=False)
        assert constraints.size == sum([c.num_constraints for c in solver.constraint_module_dict.values()])

    @staticmethod
    def test_x_to_x2(solver_class: IPOPTSolver.__class__, env_class: GraphBasedEnvironment.__class__):
        env, solver, z0 = scenario(solver_class, env_class=env_class)
        x0 = solver.z_to_ego_trajectory(z0).detach().numpy()[:, 0:2]

        x02 = np.reshape(x0, (-1, 2))
        for xi in x02:
            assert any([np.isclose(np.linalg.norm(xk - xi), 0, atol=2.0) for xk in x0])

    @staticmethod
    def test_runtime(solver_class: IPOPTSolver.__class__, env_class: GraphBasedEnvironment.__class__):
        env, solver, z0 = scenario(solver_class, env_class=env_class)

        comp_times_objective = []
        comp_times_constraints = []
        for _ in range(10):
            start_time = time.time()
            solver.constraints(z=z0, tag="core0")
            comp_times_constraints.append(time.time() - start_time)
            start_time = time.time()
            solver.objective(z=z0, tag="core0")
            comp_times_objective.append(time.time() - start_time)

        assert np.mean(comp_times_objective) < 0.04  # faster than 25 Hz (!)
        assert np.mean(comp_times_constraints) < 0.04  # faster than 25 Hz (!)

    @staticmethod
    def test_solve(solver_class: IPOPTSolver.__class__, env_class: GraphBasedEnvironment.__class__):
        env = env_class(IntegratorDTAgent, {"position": torch.tensor([-8, 0])})
        env.add_ado(position=torch.tensor([0, 0]), velocity=torch.tensor([-1, 0]))
        solver = solver_class(env, goal=torch.zeros(2), t_planning=5, verbose=0, multiprocessing=False)

        assert solver.T == 5
        assert torch.all(torch.eq(solver.goal, torch.zeros(2)))

        solver_horizon = 3
        ego_trajectory_opt, ado_trajectories = solver.solve(solver_horizon, max_cpu_time=0.1)
        ado_planned = solver.log["opt/ado_planned"]
        ego_opt_planned = solver.log["opt/ego_planned"]

        # Test output shapes.
        assert check_ego_trajectory(ego_trajectory_opt, t_horizon=solver_horizon + 1)
        assert check_ado_trajectories(ado_trajectories, t_horizon=solver_horizon + 1, ados=env.num_ados, modes=1)
        assert tuple(ado_planned.shape) == (solver_horizon, 1, 1, solver.T + 1, 5)
        assert tuple(ego_opt_planned.shape) == (solver_horizon, solver.T + 1, 5)

        # Test ado planned trajectories - depending on environment engine. Therefore only time-stamps can be tested.
        time_steps_exp = torch.arange(start=env.time, end=env.time + env.dt * (solver_horizon + 1), step=env.dt)
        assert torch.all(torch.isclose(ego_trajectory_opt[:, -1], time_steps_exp))
        for k in range(solver_horizon):
            t_start = env.time + (k + 1) * env.dt
            time_steps_exp = torch.linspace(start=t_start, end=t_start + env.dt * solver.T, steps=solver.T + 1)
            assert torch.all(torch.isclose(ego_opt_planned[k, :, -1], time_steps_exp))

        # Test constraint satisfaction - minimal distance constraint.
        if CONSTRAINT_MIN_DISTANCE in solver.constraint_modules:
            limit, _ = solver.constraint_module_dict[CONSTRAINT_MIN_DISTANCE].constraint_bounds()
            for t in range(solver_horizon):
                for m in range(env.num_ghosts):
                    distance = ego_trajectory_opt[t, 0:2] - ado_trajectories[m, :, t, 0:2]
                    assert torch.all(torch.ge(torch.norm(distance, dim=1), CONSTRAINT_MIN_L2_DISTANCE))

        # Test constraint satisfaction - maximal speed constraint.
        if CONSTRAINT_MAX_SPEED in solver.constraint_modules:
            _, limit = solver.constraint_module_dict[CONSTRAINT_MAX_SPEED].constraint_bounds()
            speeds = torch.norm(ego_trajectory_opt[:, 2:4], dim=1).detach().numpy()
            assert torch.all(torch.le(speeds, AGENT_SPEED_MAX))


###########################################################################
# Test - IPOPT Solver #####################################################
###########################################################################
@pytest.mark.parametrize("solver_class", [IGradSolver, SGradSolver])
@pytest.mark.parametrize("env_class", ENVIRONMENTS)
class TestIPOPTSolvers:

    @staticmethod
    def test_formulation(solver_class: IPOPTSolver.__class__, env_class: GraphBasedEnvironment.__class__):
        env, solver, z0 = scenario(solver_class, env_class=env_class)

        # Test output shapes.
        grad = solver.gradient(z=z0)
        assert np.linalg.norm(grad) > 0
        assert grad.size == z0.flatten().size
        jacobian = solver.jacobian(z0)
        num_constraints = sum([c.num_constraints for c in solver.constraint_module_dict.values()])
        assert jacobian.size == num_constraints * z0.flatten().size

    @staticmethod
    def test_runtime(solver_class: IPOPTSolver.__class__, env_class: GraphBasedEnvironment.__class__):
        env, solver, z0 = scenario(solver_class, env_class=env_class)

        comp_times_jacobian = []
        comp_times_gradient = []
        for _ in range(10):
            start_time = time.time()
            solver.gradient(z=z0)
            comp_times_gradient.append(time.time() - start_time)
            start_time = time.time()
            solver.jacobian(z=z0)
            comp_times_jacobian.append(time.time() - start_time)
        assert np.mean(comp_times_jacobian) < 0.07  # faster than 15 Hz (!)
        assert np.mean(comp_times_gradient) < 0.07  # faster than 15 Hz (!)
