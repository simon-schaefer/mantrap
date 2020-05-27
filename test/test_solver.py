import math

import numpy as np
import pytest
import torch

import mantrap.constants
import mantrap.agents
import mantrap.environment
import mantrap.attention
import mantrap.modules
import mantrap.solver
import mantrap.utility.shaping

environments = [mantrap.environment.KalmanEnvironment,
                mantrap.environment.PotentialFieldEnvironment,
                mantrap.environment.SocialForcesEnvironment,
                mantrap.environment.Trajectron]
attentions = [mantrap.attention.EuclideanModule,
              mantrap.attention.ReachabilityModule]


def scenario(
    solver_class: mantrap.solver.base.TrajOptSolver.__class__,
    env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
    num_modes: int = 1,
    attention_class: str = None,
    **solver_kwargs
):
    env = env_class(mantrap.agents.IntegratorDTAgent, ego_position=torch.tensor([-5, 2]))
    if num_modes > 1 and not env.is_multi_modal:
        pytest.skip()
    env.add_ado(position=torch.tensor([3, 2]), num_modes=num_modes)
    solver = solver_class(env, goal=torch.tensor([1, 1]), attention_module=attention_class, **solver_kwargs)
    z0 = solver.warm_start().detach().numpy()
    return env, solver, z0


###########################################################################
# Tests - All Solvers #####################################################
###########################################################################
@pytest.mark.parametrize("solver_class", [mantrap.solver.SGradSolver,
                                          mantrap.solver.MonteCarloTreeSearch,
                                          mantrap.solver.baselines.RandomSearch,
                                          mantrap.solver.baselines.ORCASolver])
@pytest.mark.parametrize("env_class", environments)
@pytest.mark.parametrize("num_modes", [1, 3])
@pytest.mark.parametrize("attention_class", attentions)
class TestSolvers:

    @staticmethod
    def test_convergence(solver_class: mantrap.solver.base.TrajOptSolver.__class__,
                         env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
                         num_modes: int,
                         attention_class: mantrap.attention.AttentionModule.__class__):
        dt = mantrap.constants.ENV_DT_DEFAULT
        ego_goal_distance = (mantrap.constants.PED_SPEED_MAX / 2) * dt
        env = env_class(mantrap.agents.IntegratorDTAgent, ego_position=torch.tensor([-ego_goal_distance, 0]), dt=dt)
        env.add_ado(position=torch.ones(2) * 10, velocity=torch.zeros(2))

        solver_kwargs = {"attention_module": attention_class, "t_planning": 1}
        solver = solver_class(env, goal=torch.zeros(2), objectives=[("goal", 1.0)], constraints=[], **solver_kwargs)

        z0 = solver.warm_start()
        z_opt, _, _ = solver.optimize(z0=z0, tag="core0", max_cpu_time=1.0)
        ego_controls = solver.z_to_ego_controls(z=z_opt.detach().numpy())
        ego_trajectory_opt = solver.env.ego.unroll_trajectory(controls=ego_controls, dt=solver.env.dt)

        assert torch.all(torch.isclose(ego_trajectory_opt[0, :], env.ego.state_with_time))
        for k in range(1, solver.planning_horizon):
            assert torch.all(torch.isclose(ego_trajectory_opt[k, 0:2], solver.goal, atol=0.5))

    @staticmethod
    def test_formulation(solver_class: mantrap.solver.base.TrajOptSolver.__class__,
                         env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
                         num_modes: int,
                         attention_class: mantrap.attention.AttentionModule.__class__):
        env, solver, z0 = scenario(solver_class, env_class=env_class, num_modes=num_modes,
                                   attention_class=attention_class)

        # Test output shapes.
        objective = solver.objective(z=z0, tag="core0")
        assert type(objective) == float
        constraints = solver.constraints(z=z0, tag="core0", return_violation=False)
        assert constraints.size == sum([c.num_constraints(ado_ids=env.ado_ids) for c in solver.modules])

    @staticmethod
    def test_z_to_ego_trajectory(solver_class: mantrap.solver.base.TrajOptSolver.__class__,
                                 env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
                                 num_modes: int,
                                 attention_class: mantrap.attention.AttentionModule.__class__):
        env, solver, z0 = scenario(solver_class, env_class, num_modes=num_modes, attention_class=attention_class)
        ego_trajectory_initial = solver.z_to_ego_trajectory(z0).detach().numpy()[:, 0:2]

        x02 = np.reshape(ego_trajectory_initial, (-1, 2))
        for xi in x02:
            assert any([np.isclose(np.linalg.norm(xk - xi), 0, atol=2.0) for xk in ego_trajectory_initial])

    @staticmethod
    def test_num_optimization_variables(solver_class: mantrap.solver.base.TrajOptSolver.__class__,
                                        env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
                                        num_modes: int,
                                        attention_class: mantrap.attention.AttentionModule.__class__):
        _, solver, _ = scenario(solver_class, env_class=env_class, num_modes=num_modes, attention_class=attention_class)
        lb, ub = solver.optimization_variable_bounds()

        assert len(lb) == solver.num_optimization_variables()
        assert len(ub) == solver.num_optimization_variables()

    @staticmethod
    def test_solve(solver_class: mantrap.solver.base.TrajOptSolver.__class__,
                   env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
                   num_modes: int,
                   attention_class: mantrap.attention.AttentionModule.__class__):
        env = env_class(mantrap.agents.IntegratorDTAgent, ego_position=torch.tensor([-8, 0]))
        if num_modes > 1 and not env.is_multi_modal:
            pytest.skip()
        env.add_ado(position=torch.tensor([0, 0]), velocity=torch.tensor([-1, 0]), num_modes=num_modes)
        solver = solver_class(env, attention_module=attention_class, goal=torch.zeros(2), t_planning=5)

        assert solver.planning_horizon == 5
        assert torch.all(torch.eq(solver.goal, torch.zeros(2)))

        solver_horizon = 3
        ego_trajectory_opt, ado_trajectories = solver.solve(solver_horizon, max_cpu_time=0.1)
        ado_planned = solver.log[f"{mantrap.constants.TAG_OPTIMIZATION}/ado_planned_end"]
        ego_opt_planned = solver.log[f"{mantrap.constants.TAG_OPTIMIZATION}/ego_planned_end"]

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
            t_start = env.time + k * env.dt
            time_steps_exp = torch.linspace(start=t_start,
                                            end=t_start + env.dt * solver.planning_horizon,
                                            steps=solver.planning_horizon + 1)
            assert torch.all(torch.isclose(ego_opt_planned[k, :, -1], time_steps_exp))

        # Test constraint satisfaction - automatic constraint violation test. The constraints have to be met for
        # the optimized trajectory, therefore the violation has to be zero (i.e. all constraints are not active).
        # Since the constraint modules have been tested independently, for generalization, the module-internal
        # violation computation can be used for this check.
        for module in solver.modules:
            violation = module.compute_violation(ego_trajectory_opt, ado_ids=env.ado_ids, tag="test")
            assert math.isclose(violation, 0.0, abs_tol=1e-3)

    @staticmethod
    def test_warm_start(solver_class: mantrap.solver.base.TrajOptSolver.__class__,
                        env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
                        num_modes: int,
                        attention_class: mantrap.attention.AttentionModule.__class__):
        env = env_class(mantrap.agents.IntegratorDTAgent,
                        ego_position=torch.tensor([-8, 0]),
                        ego_velocity=torch.ones(2))
        if num_modes > 1 and not env.is_multi_modal:
            pytest.skip()
        env.add_ado(position=torch.tensor([0, 0]), velocity=torch.tensor([-1, 0]), num_modes=num_modes)
        solver = solver_class(env, attention_module=attention_class, goal=torch.zeros(2), t_planning=5)

        # Solve simplified optimization problem.
        z0 = solver.warm_start().detach().numpy()

        # Check whether z is within the allowed optimization boundaries.
        z0_flat = z0.flatten()
        lower, upper = solver.optimization_variable_bounds()
        assert z0_flat.size == len(lower) == len(upper)
        assert np.all(np.less_equal(z0_flat, upper))
        assert np.all(np.greater_equal(z0_flat, lower))


###########################################################################
# Test - Search Solver ####################################################
###########################################################################
@pytest.mark.parametrize("solver_class", [mantrap.solver.baselines.RandomSearch,
                                          mantrap.solver.MonteCarloTreeSearch])
@pytest.mark.parametrize("env_class", environments)
class TestSearchSolvers:

    @staticmethod
    def test_improvement(solver_class: mantrap.solver.base.TrajOptSolver.__class__,
                         env_class: mantrap.environment.base.GraphBasedEnvironment.__class__):
        env = env_class(mantrap.agents.IntegratorDTAgent, ego_position=torch.tensor([-8, 0]))
        env.add_ado(position=torch.tensor([9, 9]))  # far-away
        solver = solver_class(env, goal=torch.zeros(2), t_planning=5)

        z0 = np.random.uniform(*solver.z_bounds)
        obj_0, _ = solver.evaluate(z0, ado_ids=env.ado_ids, tag="")
        _, obj_best, _ = solver.optimize_core(z0, ado_ids=env.ado_ids)

        assert obj_0 >= obj_best


###########################################################################
# Test - IPOPT Solver #####################################################
###########################################################################
@pytest.mark.parametrize("solver_class", [mantrap.solver.SGradSolver])
@pytest.mark.parametrize("env_class", environments)
@pytest.mark.parametrize("num_modes", [1, 3])
@pytest.mark.parametrize("attention_class", attentions)
class TestIPOPTSolvers:

    @staticmethod
    def test_formulation(solver_class: mantrap.solver.base.TrajOptSolver.__class__,
                         env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
                         num_modes: int,
                         attention_class: mantrap.attention.AttentionModule.__class__):
        env, solver, z0 = scenario(solver_class, env_class=env_class, num_modes=num_modes,
                                   attention_class=attention_class)

        # Test output shapes.
        grad = solver.gradient(z=z0)
        assert np.linalg.norm(grad) > 0
        assert grad.size == z0.flatten().size

        jacobian = solver.jacobian(z0)
        num_constraints = sum([c.num_constraints(ado_ids=env.ado_ids) for c in solver.modules])

        # Jacobian is only defined if the environment
        if all([module.gradient_condition() for module in solver.modules]):
            assert jacobian.size == num_constraints * z0.size
        else:
            print(jacobian.size, num_constraints * z0.size)
            assert jacobian.size <= num_constraints * z0.size


@pytest.mark.parametrize("env_class", environments)
def test_ignoring_solver(env_class: mantrap.environment.base.GraphBasedEnvironment.__class__):
    ego_position = torch.tensor([-3, 0])
    ego_velocity = torch.ones(2)
    env = env_class(mantrap.agents.DoubleIntegratorDTAgent,
                    ego_position=ego_position,
                    ego_velocity=ego_velocity,
                    dt=0.4)
    env.add_ado(position=torch.zeros(2), velocity=torch.zeros(2))

    modules = [(mantrap.modules.GoalNormModule, {"optimize_speed": False}),
               (mantrap.modules.ControlLimitModule, None)]

    solver = mantrap.solver.SGradSolver(env, goal=torch.tensor([1, 0]), t_planning=3, modules=modules)
    ego_trajectory, _ = solver.solve(time_steps=20)

    # Check whether goal has been reached in acceptable closeness.
    goal_distance = torch.norm(ego_trajectory[-1,  0:2] - solver.goal)
    assert torch.le(goal_distance, mantrap.constants.SOLVER_GOAL_END_DISTANCE * 2)
