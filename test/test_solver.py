import math

import numpy as np
import pytest
import torch

import mantrap.constants
import mantrap.agents
import mantrap.environment
import mantrap.filter
import mantrap.modules
import mantrap.solver
import mantrap.utility.shaping

environments = [mantrap.environment.KalmanEnvironment,
                mantrap.environment.PotentialFieldEnvironment,
                mantrap.environment.SocialForcesEnvironment,
                mantrap.environment.ORCAEnvironment,
                mantrap.environment.Trajectron]
filters = [mantrap.filter.EuclideanModule,
           mantrap.filter.ReachabilityModule]


def scenario(
    solver_class: mantrap.solver.base.TrajOptSolver.__class__,
    env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
    num_modes: int = 1,
    filter_class: str = None,
    **solver_kwargs
):
    env = env_class(mantrap.agents.IntegratorDTAgent, {"position": torch.tensor([-5, 2])})
    if num_modes > 1 and not env.is_multi_modal:
        pytest.skip()
    env.add_ado(position=torch.tensor([3, 2]), num_modes=num_modes)
    solver = solver_class(env, goal=torch.tensor([1, 1]), filter_module=filter_class, **solver_kwargs)
    z0 = solver.initial_values(just_one=True).detach().numpy()
    return env, solver, z0


###########################################################################
# Tests - All Solvers #####################################################
###########################################################################
@pytest.mark.parametrize("solver_class", [mantrap.solver.SGradSolver,
                                          mantrap.solver.MonteCarloTreeSearch,
                                          mantrap.solver.ORCASolver])
@pytest.mark.parametrize("env_class", environments)
@pytest.mark.parametrize("num_modes", [1, 3])
@pytest.mark.parametrize("filter_class", filters)
class TestSolvers:

    @staticmethod
    def test_convergence(solver_class, env_class, num_modes, filter_class):
        dt = mantrap.constants.ENV_DT_DEFAULT
        ego_goal_distance = (mantrap.constants.AGENT_SPEED_MAX / 2) * dt
        env = env_class(mantrap.agents.IntegratorDTAgent, {"position": torch.tensor([-ego_goal_distance, 0])}, dt=dt)
        env.add_ado(position=torch.ones(2) * 10, velocity=torch.zeros(2))

        solver_kwargs = {"filter_module": filter_class, "t_planning": 1}
        solver = solver_class(env, goal=torch.zeros(2), objectives=[("goal", 1.0)], constraints=[], **solver_kwargs)

        z0 = solver.initial_values(just_one=True)
        z_opt, _, _ = solver.optimize(z0=z0, tag="core0", max_cpu_time=1.0, max_iter=1000)
        ego_controls = solver.z_to_ego_controls(z=z_opt.detach().numpy())
        ego_trajectory_opt = solver.env.ego.unroll_trajectory(controls=ego_controls, dt=solver.env.dt)

        assert torch.all(torch.isclose(ego_trajectory_opt[0, :], env.ego.state_with_time))
        for k in range(1, solver.planning_horizon):
            assert torch.all(torch.isclose(ego_trajectory_opt[k, 0:2], solver.goal, atol=0.5))

    @staticmethod
    def test_formulation(solver_class, env_class, num_modes, filter_class):
        env, solver, z0 = scenario(solver_class, env_class=env_class, num_modes=num_modes, filter_class=filter_class)

        # Test output shapes.
        objective = solver.objective(z=z0, tag="core0")
        assert type(objective) == float
        constraints = solver.constraints(z=z0, tag="core0", return_violation=False)
        assert constraints.size == sum([c.num_constraints() for c in solver.modules])

    @staticmethod
    def test_z_to_ego_trajectory(solver_class, env_class, num_modes, filter_class):
        env, solver, z0 = scenario(solver_class, env_class=env_class, num_modes=num_modes, filter_class=filter_class)
        ego_trajectory_initial = solver.z_to_ego_trajectory(z0).detach().numpy()[:, 0:2]

        x02 = np.reshape(ego_trajectory_initial, (-1, 2))
        for xi in x02:
            assert any([np.isclose(np.linalg.norm(xk - xi), 0, atol=2.0) for xk in ego_trajectory_initial])

    @staticmethod
    def test_num_optimization_variables(solver_class, env_class, num_modes, filter_class):
        _, solver, _ = scenario(solver_class, env_class=env_class, num_modes=num_modes, filter_class=filter_class)
        lb, ub = solver.optimization_variable_bounds()

        assert len(lb) == solver.num_optimization_variables()
        assert len(ub) == solver.num_optimization_variables()

        z0s = solver.initial_values(just_one=False)
        assert len(z0s.shape) == 2
        print(z0s.shape)

        z0 = solver.initial_values(just_one=True)
        assert len(z0.shape) == 1

    @staticmethod
    def test_solve(solver_class, env_class, num_modes, filter_class):
        env = env_class(mantrap.agents.IntegratorDTAgent, {"position": torch.tensor([-8, 0])})
        if num_modes > 1 and not env.is_multi_modal:
            pytest.skip()
        env.add_ado(position=torch.tensor([0, 0]), velocity=torch.tensor([-1, 0]), num_modes=num_modes)
        solver = solver_class(env, filter_module=filter_class, goal=torch.zeros(2), t_planning=5)

        assert solver.planning_horizon == 5
        assert torch.all(torch.eq(solver.goal, torch.zeros(2)))

        solver_horizon = 3
        ego_trajectory_opt, ado_trajectories = solver.solve(solver_horizon, max_cpu_time=0.1, multiprocessing=False)
        ado_planned = solver.log["opt/ado_planned"]
        ego_opt_planned = solver.log["opt/ego_planned"]

        # Test output shapes.
        t_horizon_exp = solver_horizon + 1  # t_controls = solver_horizon, t_trajectory = solver_horizon + 1
        modes_exp = 1  # output path is deterministic, so uni-modal
        assert mantrap.utility.shaping.check_ego_trajectory(ego_trajectory_opt, t_horizon=t_horizon_exp)
        assert mantrap.utility.shaping.check_ado_trajectories(ado_trajectories, t_horizon_exp,
                                                              ados=env.num_ados, modes=modes_exp)
        assert tuple(ado_planned.shape) == (solver_horizon, 1, num_modes, solver.planning_horizon + 1, 5)
        assert tuple(ego_opt_planned.shape) == (solver_horizon, solver.planning_horizon + 1, 5)

        # Test ado planned trajectories - depending on environment engine. Therefore only time-stamps can be tested.
        time_steps_exp = torch.arange(start=env.time, end=env.time + env.dt * (solver_horizon + 1), step=env.dt)
        assert torch.all(torch.isclose(ego_trajectory_opt[:, -1], time_steps_exp))
        for k in range(solver_horizon):
            t_start = env.time + (k + 1) * env.dt
            time_steps_exp = torch.linspace(start=t_start,
                                            end=t_start + env.dt * solver.planning_horizon,
                                            steps=solver.planning_horizon + 1)
            assert torch.all(torch.isclose(ego_opt_planned[k, :, -1], time_steps_exp))

        # Test constraint satisfaction - automatic constraint violation test. The constraints have to be met for
        # the optimized trajectory, therefore the violation has to be zero (i.e. all constraints are not active).
        # Since the constraint modules have been tested independently, for generalization, the module-internal
        # violation computation can be used for this check.
        for module in solver.modules:
            violation = module.compute_violation(ego_trajectory_opt, ado_ids=None)
            assert math.isclose(violation, 0.0, abs_tol=1e-3)


###########################################################################
# Test - IPOPT Solver #####################################################
###########################################################################
@pytest.mark.parametrize("solver_class", [mantrap.solver.SGradSolver])
@pytest.mark.parametrize("env_class", environments)
@pytest.mark.parametrize("num_modes", [1, 3])
@pytest.mark.parametrize("filter_class", filters)
class TestIPOPTSolvers:

    @staticmethod
    def test_formulation(solver_class, env_class, num_modes, filter_class):
        env, solver, z0 = scenario(solver_class, env_class=env_class, num_modes=num_modes, filter_class=filter_class)

        # Test output shapes.
        grad = solver.gradient(z=z0)
        assert np.linalg.norm(grad) > 0
        assert grad.size == z0.flatten().size

        jacobian = solver.jacobian(z0)
        num_constraints = sum([c.num_constraints() for c in solver.modules])

        # Jacobian is only defined if the environment
        if all([module._gradient_condition() for module in solver.modules]):
            assert jacobian.size == num_constraints * z0.size
        else:
            print(jacobian.size, num_constraints * z0.size)
            assert jacobian.size <= num_constraints * z0.size


@pytest.mark.parametrize("env_class", environments)
def test_ignoring_solver(env_class):
    ego_position = torch.tensor([-3, 0])
    ego_velocity = torch.ones(2)
    env = env_class(mantrap.agents.DoubleIntegratorDTAgent,
                    {"position": ego_position, "velocity": ego_velocity},
                    dt=0.4)
    env.add_ado(position=torch.zeros(2), velocity=torch.zeros(2))

    modules = [(mantrap.modules.GoalModule, {"optimize_speed": False}),
               (mantrap.modules.ControlLimitModule, None)]

    solver = mantrap.solver.SGradSolver(env, goal=torch.tensor([1, 0]), t_planning=3, modules=modules)
    ego_trajectory, _ = solver.solve(time_steps=20, multiprocessing=False)

    # Check whether goal has been reached in acceptable closeness.
    goal_distance = torch.norm(ego_trajectory[-1,  0:2] - solver.goal)
    assert torch.le(goal_distance, mantrap.constants.SOLVER_GOAL_END_DISTANCE * 2)
