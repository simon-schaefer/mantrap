import math
import time

import numpy as np
import pytest
import torch

import mantrap.agents
import mantrap.environment
import mantrap.attention
import mantrap.modules
import mantrap.utility.maths


environments = [mantrap.environment.KalmanEnvironment,
                mantrap.environment.PotentialFieldEnvironment,
                mantrap.environment.SocialForcesEnvironment,
                mantrap.environment.Trajectron]


def create_scene(module_class: mantrap.modules.base.OptimizationModule.__class__,
                 env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
                 num_modes: int):
    env = env_class(mantrap.agents.DoubleIntegratorDTAgent, ego_position=torch.tensor([-5, 0.1]))
    if num_modes > 1 and not env.is_multi_modal:
        pytest.skip()
    env.add_ado(position=torch.zeros(2), goal=torch.rand(2) * 10, num_modes=num_modes)
    module = module_class(env=env, t_horizon=5, goal=torch.rand(2))  # must be general here for all modules ...
    return module, env


###########################################################################
# Optimization Module #####################################################
###########################################################################
@pytest.mark.parametrize("module_class", [mantrap.modules.InteractionProbabilityModule,
                                          mantrap.modules.baselines.InteractionPositionModule,
                                          mantrap.modules.baselines.InteractionAccelerationModule,
                                          mantrap.modules.GoalNormModule,
                                          mantrap.modules.baselines.GoalSumModule,

                                          mantrap.modules.ControlLimitModule,
                                          mantrap.modules.baselines.MinDistanceModule,
                                          mantrap.modules.HJReachabilityModule,
                                          mantrap.modules.SpeedLimitModule,

                                          mantrap.attention.EuclideanModule,
                                          mantrap.attention.ReachabilityModule,
                                          ])
@pytest.mark.parametrize("env_class", environments)
@pytest.mark.parametrize("num_modes", [1, 2])
class TestOptimizationModules:

    @staticmethod
    def test_internal_env_update(module_class: mantrap.modules.base.OptimizationModule.__class__,
                                 env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
                                 num_modes: int):
        module, env = create_scene(module_class, env_class=env_class, num_modes=num_modes)

        # Compare the environment with the module-internally's environment states.
        assert torch.all(torch.eq(module.env.states()[0], env.states()[0]))
        assert torch.all(torch.eq(module.env.states()[1], env.states()[1]))

        # Add agent to environment (i.e. change the environment) and check again.
        env.add_ado(position=torch.tensor([5, 1]), goal=torch.rand(2) * (-10), num_modes=num_modes)
        assert torch.all(torch.eq(module.env.states()[0], env.states()[0]))
        assert torch.all(torch.eq(module.env.states()[1], env.states()[1]))

        # Step environment (i.e. change environment internally) and check again.
        env.step(ego_action=torch.rand(2))
        assert torch.all(torch.eq(module.env.states()[0], env.states()[0]))
        assert torch.all(torch.eq(module.env.states()[1], env.states()[1]))


###########################################################################
# Objectives ##############################################################
###########################################################################
@pytest.mark.parametrize("module_class", [mantrap.modules.InteractionProbabilityModule,
                                          mantrap.modules.baselines.InteractionPositionModule,
                                          mantrap.modules.baselines.InteractionAccelerationModule,
                                          mantrap.modules.GoalNormModule,
                                          mantrap.modules.baselines.GoalSumModule])
@pytest.mark.parametrize("env_class", environments)
@pytest.mark.parametrize("num_modes", [1, 2])
class TestObjectives:

    @staticmethod
    def test_gradient_analytical(module_class: mantrap.modules.base.OptimizationModule.__class__,
                                 env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
                                 num_modes: int):
        env = env_class(mantrap.agents.DoubleIntegratorDTAgent, ego_position=torch.rand(2))
        if num_modes > 1 and not env.is_multi_modal:
            pytest.skip()
        env.add_ado(position=torch.rand(2) * 5, goal=torch.rand(2) * 10, num_modes=num_modes)
        env.add_ado(position=torch.rand(2) * 8, goal=torch.rand(2) * (-10), num_modes=num_modes)

        t_horizon = 10
        ego_controls = torch.rand((t_horizon, 2)) / 10.0
        ego_controls.requires_grad = True
        ego_trajectory = env.ego.unroll_trajectory(controls=ego_controls, dt=env.dt)

        # Compute analytical gradient, if it is not defined (= returning `None`) just skip this
        # test since there is nothing to test here anymore.
        module = module_class(goal=torch.rand(2) * 8, env=env, t_horizon=t_horizon)
        gradient_analytical = module.compute_gradient_analytically(ego_trajectory=ego_trajectory,
                                                                   grad_wrt=ego_controls,
                                                                   ado_ids=env.ado_ids,
                                                                   tag="test")

        if gradient_analytical is None:
            pytest.skip()

        # Otherwise compute jacobian "numerically", i.e. using the PyTorch autograd module.
        # Then assert equality (or numerical equality) between both results.
        objective = module.objective_core(ego_trajectory, ado_ids=env.ado_ids, tag="test")
        gradient_auto_grad = module.compute_gradient_auto_grad(objective, grad_wrt=ego_controls)
        assert np.allclose(gradient_analytical, gradient_auto_grad, atol=0.01)

    @staticmethod
    def test_multimodal_support(module_class: mantrap.modules.base.OptimizationModule.__class__,
                                env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
                                num_modes: int):
        env = env_class(mantrap.agents.IntegratorDTAgent, ego_position=torch.tensor([-5, 0.1]))
        if num_modes > 1 and not env.is_multi_modal:
            pytest.skip()
        env.add_ado(position=torch.zeros(2),
                    velocity=torch.tensor([-1, 0]),
                    num_modes=num_modes,
                    goal=torch.tensor([-5, 0])
                    )
        ego_trajectory = env.ego.unroll_trajectory(controls=torch.ones((10, 2)), dt=env.dt)

        module = module_class(t_horizon=10, env=env, goal=torch.rand(2))
        assert module.objective(ego_trajectory, ado_ids=env.ado_ids, tag="test") is not None

    @staticmethod
    def test_output(module_class: mantrap.modules.base.OptimizationModule.__class__,
                    env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
                    num_modes: int):
        env = env_class(mantrap.agents.IntegratorDTAgent, ego_position=torch.tensor([-5, 0.1]))
        if num_modes > 1 and not env.is_multi_modal:
            pytest.skip()
        env.add_ado(position=torch.zeros(2), num_modes=num_modes)
        ego_trajectory = env.ego.unroll_trajectory(controls=torch.ones((10, 2)), dt=env.dt)
        ego_trajectory.requires_grad = True

        module = module_class(t_horizon=10, env=env, goal=torch.rand(2))
        objective = module.objective(ego_trajectory, ado_ids=env.ado_ids, tag="test")
        gradient = module.gradient(ego_trajectory, grad_wrt=ego_trajectory, ado_ids=env.ado_ids, tag="test")
        assert type(objective) == float
        assert gradient.size == ego_trajectory.numel()

    @staticmethod
    def test_runtime(module_class: mantrap.modules.base.OptimizationModule.__class__,
                     env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
                     num_modes: int):
        if module_class in mantrap.modules.baselines.__dict__.values():
            pytest.skip()  # skip baseline modules
        module, env = create_scene(module_class, env_class=env_class, num_modes=num_modes)

        ego_controls = torch.ones((5, 2)) / 10.0
        ego_controls.requires_grad = True
        ego_trajectory = env.ego.unroll_trajectory(controls=ego_controls, dt=env.dt)

        module = module_class(t_horizon=5, env=env, goal=torch.rand(2))
        objective_run_times, gradient_run_times = list(), list()
        for i in range(30):
            start_time = time.time()
            module.objective(ego_trajectory, ado_ids=env.ado_ids, tag="test")
            objective_run_times.append(time.time() - start_time)

            start_time = time.time()
            module.gradient(ego_trajectory, grad_wrt=ego_controls, ado_ids=env.ado_ids, tag="test")
            gradient_run_times.append(time.time() - start_time)

        assert np.mean(objective_run_times) < 0.02 * num_modes  # 50 Hz
        assert np.mean(gradient_run_times) < 0.03 * num_modes  # 33 Hz


@pytest.mark.parametrize("module_class", [mantrap.modules.InteractionProbabilityModule,
                                          mantrap.modules.baselines.InteractionPositionModule,
                                          mantrap.modules.baselines.InteractionAccelerationModule])
@pytest.mark.parametrize("env_class", environments)
@pytest.mark.parametrize("num_modes", [1, 2])
class TestObjectiveInteraction:

    @staticmethod
    def test_far_and_near(module_class: mantrap.modules.base.OptimizationModule.__class__,
                          env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
                          num_modes: int):
        """Every interaction-based objective should be larger the closer the interacting agents are, so having the
        ego agent close to some ado should affect the ado more than when the ego agent is far away. """
        env = env_class(mantrap.agents.IntegratorDTAgent, ego_position=torch.tensor([-5, 100.0]), y_axis=(-100, 100))
        if num_modes > 1 and not env.is_multi_modal:
            pytest.skip()
        env.add_ado(position=torch.zeros(2), num_modes=num_modes)

        start_near, end_near = torch.tensor([-5, 0.1]), torch.tensor([5, 0.1])
        ego_path_near = mantrap.utility.maths.straight_line(start=start_near, end=end_near, steps=11)
        ego_trajectory_near = env.ego.expand_trajectory(ego_path_near, dt=env.dt)

        start_far, end_far = torch.tensor([-5, 100.0]), torch.tensor([5, 10.0])
        ego_path_far = mantrap.utility.maths.straight_line(start=start_far, end=end_far, steps=11)
        ego_trajectory_far = env.ego.expand_trajectory(ego_path_far, dt=env.dt)

        module = module_class(t_horizon=10, env=env)
        if env.is_deterministic:
            objective_near = module.objective(ego_trajectory_near, ado_ids=[], tag="test")
            objective_far = module.objective(ego_trajectory_far, ado_ids=[], tag="test")
            assert objective_near >= objective_far


def test_objective_goal_distribution():
    env = mantrap.environment.PotentialFieldEnvironment(mantrap.agents.IntegratorDTAgent, ego_position=torch.zeros(2))
    goal_state = torch.tensor([4.1, 8.9])
    ego_trajectory = torch.rand((11, 4))

    module = mantrap.modules.GoalNormModule(goal=goal_state, env=env, t_horizon=10, weight=1.0)
    objective = module.objective(ego_trajectory, ado_ids=[], tag="test")
    distance = float(torch.mean(torch.sum((ego_trajectory[:, 0:2] - goal_state).pow(2), dim=1)).item())

    assert math.isclose(objective, distance, abs_tol=0.1)


###########################################################################
# Constraints #############################################################
###########################################################################
@pytest.mark.parametrize("module_class", [mantrap.modules.ControlLimitModule,
                                          mantrap.modules.baselines.MinDistanceModule,
                                          mantrap.modules.HJReachabilityModule,
                                          mantrap.modules.SpeedLimitModule])
@pytest.mark.parametrize("env_class", [mantrap.environment.KalmanEnvironment,
                                       mantrap.environment.PotentialFieldEnvironment,
                                       mantrap.environment.SocialForcesEnvironment,
                                       mantrap.environment.ORCAEnvironment,
                                       mantrap.environment.Trajectron])
@pytest.mark.parametrize("num_modes", [1, 2])
class TestConstraints:

    @staticmethod
    def test_runtime(module_class: mantrap.modules.base.OptimizationModule.__class__,
                     env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
                     num_modes: int):
        if module_class in mantrap.modules.baselines.__dict__.values():
            pytest.skip()  # skip baseline modules

        env = env_class(mantrap.agents.DoubleIntegratorDTAgent, ego_position=torch.tensor([-5, 0.1]))
        if num_modes > 1 and not env.is_multi_modal:
            pytest.skip()
        env.add_ado(position=torch.zeros(2), goal=torch.rand(2) * 10, num_modes=num_modes)
        env.add_ado(position=torch.tensor([5, 1]), goal=torch.rand(2) * (-10), num_modes=num_modes)

        ego_controls = torch.ones((5, 2)) / 10.0
        ego_controls.requires_grad = True
        ego_trajectory = env.ego.unroll_trajectory(controls=ego_controls, dt=env.dt)

        module = module_class(env=env, t_horizon=5)
        constraint_run_times, jacobian_run_times = list(), list()
        for i in range(30):
            start_time = time.time()
            module.constraint(ego_trajectory, ado_ids=env.ado_ids, tag="test")
            constraint_run_times.append(time.time() - start_time)

            start_time = time.time()
            module.jacobian(ego_trajectory, grad_wrt=ego_controls, ado_ids=env.ado_ids, tag="test")
            jacobian_run_times.append(time.time() - start_time)

        assert np.mean(constraint_run_times) < 0.01 * num_modes  # 100 Hz
        assert np.mean(jacobian_run_times) < 0.02 * num_modes  # 50 Hz

    @staticmethod
    def test_violation(module_class: mantrap.modules.base.OptimizationModule.__class__,
                       env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
                       num_modes: int):
        """In order to test the constraint violation in general test it in a scene with static and far-distant
        agent(s), with  respect to the ego, and static ego robot. In this configurations all constraints should
        be met. """
        env = env_class(mantrap.agents.DoubleIntegratorDTAgent, ego_position=torch.tensor([-5, 0.1]))
        if num_modes > 1 and not env.is_multi_modal:
            pytest.skip()
        env.add_ado(position=torch.ones(2) * 9, goal=torch.ones(2) * 9, num_modes=num_modes)
        ego_trajectory = env.ego.unroll_trajectory(controls=torch.zeros((1, 2)), dt=env.dt)

        module = module_class(env=env, t_horizon=1)
        violation = module.compute_violation(ego_trajectory=ego_trajectory, ado_ids=env.ado_ids, tag="test")
        assert violation == 0

    @staticmethod
    def test_jacobian_analytical(module_class: mantrap.modules.base.OptimizationModule.__class__,
                                 env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
                                 num_modes: int):
        if num_modes > 1:
            pytest.skip()

        env = env_class(mantrap.agents.DoubleIntegratorDTAgent, ego_position=torch.rand(2))
        env.add_ado(position=torch.rand(2) * 5, goal=torch.rand(2) * 10, num_modes=1)
        env.add_ado(position=torch.rand(2) * 8, goal=torch.rand(2) * (-10), num_modes=1)

        ego_controls = torch.rand((5, 2)) / 10.0
        ego_controls.requires_grad = True
        ego_trajectory = env.ego.unroll_trajectory(controls=ego_controls, dt=env.dt)

        # Compute analytical jacobian, if it is not defined (= returning `None`) just skip this
        # test since there is nothing to test here anymore.
        module = module_class(env=env, t_horizon=5)
        jacobian_analytical = module.compute_jacobian_analytically(ego_trajectory=ego_trajectory,
                                                                   grad_wrt=ego_controls,
                                                                   ado_ids=env.ado_ids,
                                                                   tag="test")

        if jacobian_analytical is None or not module.gradient_condition():
            pytest.skip()

        # Otherwise compute jacobian "numerically", i.e. using the PyTorch autograd module.
        # Then assert equality (or numerical equality) between both results.
        constraints = module.constraint_core(ego_trajectory, ado_ids=env.ado_ids, tag="test")
        jacobian_auto_grad = module.compute_gradient_auto_grad(constraints, grad_wrt=ego_controls)
        assert np.allclose(jacobian_analytical, jacobian_auto_grad, atol=0.01)


@pytest.mark.parametrize("env_class", environments)
@pytest.mark.parametrize("num_modes", [1, 2])
def test_control_limit_violation(env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
                                 num_modes: int):
    position, velocity = torch.tensor([-5, 0.1]), torch.zeros(2)
    env = env_class(mantrap.agents.IntegratorDTAgent, ego_position=position, ego_velocity=velocity)
    if num_modes > 1 and not env.is_multi_modal:
        pytest.skip()

    module = mantrap.modules.ControlLimitModule(env=env, t_horizon=5)
    _, upper_bound = module.constraint_limits()

    # In this first scenario the ego has zero velocity over the full horizon.
    controls = torch.zeros((module.t_horizon, 2))
    ego_trajectory = env.ego.unroll_trajectory(controls=controls, dt=env.dt)
    violation = module.compute_violation(ego_trajectory=ego_trajectory, ado_ids=[], tag="test")
    assert violation == 0

    # In this second scenario the ego has non-zero random velocity in x-direction, but always below the maximal
    # allowed speed (random -> [0, 1]).
    controls[:, 0] = torch.rand(module.t_horizon) * upper_bound  # single integrator, so no velocity summation !
    ego_trajectory = env.ego.unroll_trajectory(controls=controls, dt=env.dt)
    violation = module.compute_violation(ego_trajectory=ego_trajectory, ado_ids=[], tag="test2")
    assert violation == 0

    # In this third scenario the ego has the same random velocity in the x-direction as in the second scenario,
    # but at one time step, it is increased to a slightly larger speed than allowed.
    controls[1, 0] = upper_bound + 1e-3
    ego_trajectory = env.ego.unroll_trajectory(controls=controls, dt=env.dt)
    violation = module.compute_violation(ego_trajectory=ego_trajectory, ado_ids=[], tag="test3")
    assert violation > 0


@pytest.mark.parametrize("env_class", environments)
@pytest.mark.parametrize("num_modes", [1, 2])
def test_speed_limit_violation(env_class: mantrap.environment.base.GraphBasedEnvironment.__class__, num_modes: int):
    position, velocity = torch.tensor([-5, 0.1]), torch.zeros(2)
    env = env_class(mantrap.agents.IntegratorDTAgent, ego_position=position, ego_velocity=velocity)
    if num_modes > 1 and not env.is_multi_modal:
        pytest.skip()

    module = mantrap.modules.SpeedLimitModule(env=env, t_horizon=5)
    lower_bound, upper_bound = module.constraint_limits()

    # In this first scenario the ego has velocities at the upper or lower bound over the full horizon.
    ego_trajectory = torch.ones((module.t_horizon + 1, 5)) * upper_bound
    ego_trajectory[2, :] = torch.ones(5) * lower_bound
    violation = module.compute_violation(ego_trajectory=ego_trajectory, ado_ids=[], tag="test")
    assert violation == 0

    # In the second scenario, one of the velocities is over the bounds.
    ego_trajectory[2, 2] = upper_bound + 0.1
    violation = module.compute_violation(ego_trajectory=ego_trajectory, ado_ids=[], tag="test")
    assert np.isclose(violation, 0.1)


@pytest.mark.parametrize("env_class", [mantrap.environment.KalmanEnvironment,
                                       mantrap.environment.PotentialFieldEnvironment,
                                       mantrap.environment.SocialForcesEnvironment,
                                       mantrap.environment.ORCAEnvironment,
                                       mantrap.environment.Trajectron])
@pytest.mark.parametrize("num_modes", [1, 2])
def test_min_distance_constraint_violation(env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
                                           num_modes: int):
    position, velocity = torch.ones(2) * 9, torch.zeros(2)
    env = env_class(mantrap.agents.IntegratorDTAgent, ego_position=position, ego_velocity=velocity)
    if num_modes > 1 and not env.is_multi_modal:
        pytest.skip()
    ado_kwargs = {"goal": torch.tensor([9, -9]), "num_modes": num_modes}
    env.add_ado(position=torch.ones(2) * (-9), velocity=torch.tensor([1, 0]), **ado_kwargs)

    controls = torch.stack((torch.ones(10) * (-1), torch.zeros(10))).view(10, 2)
    ego_trajectory = env.ego.unroll_trajectory(controls=controls, dt=env.dt)

    # In this first scenario the ado and ego are moving parallel in maximal distance to each other.
    module = mantrap.modules.baselines.MinDistanceModule(env=env, t_horizon=controls.shape[0])
    lower_bound, _ = module.constraint_limits()
    violation = module.compute_violation(ego_trajectory=ego_trajectory, ado_ids=env.ado_ids, tag="test")
    assert violation == 0

    # In the second scenario add another ado agent that is starting and moving very close to the ego robot.
    ado_start_pos = env.ego.position - (lower_bound * 0.5) * torch.ones(2)
    ado_kwargs = {"goal": ado_start_pos, "num_modes": num_modes}
    env.add_ado(position=ado_start_pos, velocity=torch.zeros(2), **ado_kwargs)

    module = mantrap.modules.baselines.MinDistanceModule(env=env, t_horizon=controls.shape[0])
    lower_bound, _ = module.constraint_limits()
    violation = module.compute_violation(ego_trajectory=ego_trajectory, ado_ids=env.ado_ids, tag="test")
    assert violation > 0


###########################################################################
# Filter ##################################################################
###########################################################################
@pytest.mark.parametrize("module_class", [mantrap.attention.EuclideanModule,
                                          mantrap.attention.ReachabilityModule])
@pytest.mark.parametrize("env_class", environments)
@pytest.mark.parametrize("num_modes", [1, 2])
class TestAttention:

    @staticmethod
    def test_runtime(module_class: mantrap.attention.AttentionModule.__class__,
                     env_class: mantrap.environment.base.GraphBasedEnvironment.__class__,
                     num_modes: int):
        env = env_class(mantrap.agents.IntegratorDTAgent, ego_position=torch.tensor([-5, 0.1]))
        if num_modes > 1 and not env.is_multi_modal:
            pytest.skip()
        env.add_ado(position=torch.zeros(2), goal=torch.rand(2) * 10, num_modes=num_modes)
        env.add_ado(position=torch.tensor([5, 1]), goal=torch.rand(2) * (-10), num_modes=num_modes)

        module = module_class(env=env, t_horizon=5)
        filter_run_times = list()
        for i in range(10):
            start_time = time.time()
            module.compute()
            filter_run_times.append(time.time() - start_time)

        assert np.mean(filter_run_times) < 0.01  # 100 Hz


###########################################################################
# Reachability Module #####################################################
###########################################################################
@pytest.mark.parametrize("env_class", environments)
@pytest.mark.parametrize("num_modes", [1, 2])
class TestHJReachability:

    @staticmethod
    def test_ego_type_failing(env_class: mantrap.environment.base.GraphBasedEnvironment.__class__, num_modes: int):
        env = env_class(ego_type=mantrap.agents.IntegratorDTAgent, ego_position=torch.zeros(2))
        if num_modes > 1 and not env.is_multi_modal:
            pytest.skip()
        env.add_ado(position=torch.rand(2) * 5, num_modes=num_modes)

        # Check for module is asserting when ego has another type than double integrator, since only for
        # this type of ego the pre-computed value function is correct.
        with pytest.raises(AssertionError):
            mantrap.modules.HJReachabilityModule(env, t_horizon=5)

    @staticmethod
    def test_constraint(env_class: mantrap.environment.base.GraphBasedEnvironment.__class__, num_modes: int):
        env = env_class(ego_type=mantrap.agents.DoubleIntegratorDTAgent, ego_position=torch.zeros(2))
        if num_modes > 1 and not env.is_multi_modal:
            pytest.skip()
        env.add_ado(position=torch.rand(2) * 5, num_modes=num_modes)
        module = mantrap.modules.HJReachabilityModule(env, t_horizon=5)

        ego_controls = torch.rand((5, 2))
        ego_trajectory = env.ego.unroll_trajectory(ego_controls, dt=env.dt)
        module.constraint(ego_trajectory, ado_ids=env.ado_ids, tag="test")

    @staticmethod
    def test_jacobi_analytical(env_class: mantrap.environment.base.GraphBasedEnvironment.__class__, num_modes: int):
        """In opposite to the other optimization modules here the full jacobian cannot be computed
        automatically using PyTorch's auto_grad framework. However the "remaining" partial derivatives
        can be computed, i.e. all except of the pre-computed one, since they are based on online
        PyTorch computations and hence have a gradient function assigned to them. """
        env = env_class(mantrap.agents.DoubleIntegratorDTAgent, ego_position=torch.rand(2))
        if num_modes > 1 and not env.is_multi_modal:
            pytest.skip()
        env.add_ado(position=torch.rand(2) * 5, goal=torch.rand(2) * 10, num_modes=num_modes)

        ego_controls = torch.rand((5, 2)) / 10.0
        ego_controls.requires_grad = True
        ego_trajectory = env.ego.unroll_trajectory(controls=ego_controls, dt=env.dt)

        # Initialize HJ module and compute partial derivative dx_rel/du_robot using auto-grad.
        module = mantrap.modules.HJReachabilityModule(env=env, t_horizon=5)
        _ = module.constraint_core(ego_trajectory, ado_ids=env.ado_ids, tag="test", enable_auto_grad=True)
        dx_rel_du_auto_grad = []
        for ado_id in env.ado_ids:
            x_rel = module.x_relative[f"test/{ado_id}"]
            grad = [torch.autograd.grad(x, ego_controls, retain_graph=True)[0] for x in x_rel]
            dx_rel_du_auto_grad.append(torch.stack(grad).reshape(4, -1))
        dx_rel_du_auto_grad = torch.stack(dx_rel_du_auto_grad)

        # Compute the same partial derivative analytically, by calling the `compute_jacobian_analytically()`
        # function. Since we cannot inverse a vector (dJ/dx_rel), we can check whether the jacobian
        # computed using the pre-computed dJ/dx_rel and the auto-grad (!) dx_rel/du results in the same
        # jacobian as the result of `compute_jacobian_analytically()`, which is only the case if
        # dx_rel/du(auto-grad) = dx_rel/du(analytic) since dJ/dx has non-zero elements.
        jacobian_analytical = module.compute_jacobian_analytically(ego_trajectory, grad_wrt=ego_controls,
                                                                   ado_ids=env.ado_ids, tag="test")
        dj_dx_rel = []
        for ado_id in env.ado_ids:
            dj_dx_rel.append(module.value_gradient(x=module.x_relative[f"test/{ado_id}"]))
        jacobian_auto_grad = np.matmul(np.stack(dj_dx_rel), dx_rel_du_auto_grad)

        assert np.allclose(jacobian_analytical, jacobian_auto_grad)
