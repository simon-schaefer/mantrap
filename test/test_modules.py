import time

import numpy as np
import pytest
import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.environment import ENVIRONMENTS
from mantrap.solver.constraints import *
from mantrap.solver.filter import *
from mantrap.solver.objectives import *
from mantrap.utility.primitives import straight_line


###########################################################################
# Objectives ##############################################################
###########################################################################
@pytest.mark.parametrize("module_class", [InteractionPositionModule, InteractionAccelerationModule])
@pytest.mark.parametrize("env_class", ENVIRONMENTS)
@pytest.mark.parametrize("num_modes", [1, 2])
class TestObjectiveInteraction:

    @staticmethod
    def test_far_and_near(module_class, env_class, num_modes):
        """Every interaction-based objective should be larger the closer the interacting agents are, so having the
        ego agent close to some ado should affect the ado more than when the ego agent is far away. """
        env = env_class(IntegratorDTAgent, {"position": torch.tensor([-5, 100.0])}, y_axis=(-100, 100))
        if num_modes > 1 and not env.is_multi_modal:
            pytest.skip()
        env.add_ado(position=torch.zeros(2), num_modes=num_modes)

        ego_path_near = straight_line(start_pos=torch.tensor([-5, 0.1]), end_pos=torch.tensor([5, 0.1]), steps=11)
        ego_trajectory_near = env.ego.expand_trajectory(ego_path_near, dt=env.dt)
        ego_path_far = straight_line(start_pos=torch.tensor([-5, 100.0]), end_pos=torch.tensor([5, 10.0]), steps=11)
        ego_trajectory_far = env.ego.expand_trajectory(ego_path_far, dt=env.dt)

        module = module_class(t_horizon=10, env=env)
        if env.is_deterministic:
            assert module.objective(ego_trajectory_near) > module.objective(ego_trajectory_far)

    @staticmethod
    def test_multimodal_support(module_class, env_class, num_modes):
        weights = np.random.rand(num_modes)
        weights_normed = (weights / weights.sum()).tolist()

        env = env_class(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1])})
        if num_modes > 1 and not env.is_multi_modal:
            pytest.skip()
        env.add_ado(position=torch.zeros(2),
                    velocity=torch.tensor([-1, 0]),
                    num_modes=num_modes,
                    weights=weights_normed,
                    goal=torch.tensor([-5, 0])
                    )
        ego_trajectory = env.ego.unroll_trajectory(controls=torch.ones((10, 2)), dt=env.dt)

        module = module_class(t_horizon=10, env=env)
        assert module.objective(ego_trajectory) is not None

    @staticmethod
    def test_output(module_class, env_class, num_modes):
        env = env_class(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1])})
        if num_modes > 1 and not env.is_multi_modal:
            pytest.skip()
        env.add_ado(position=torch.zeros(2), num_modes=num_modes)
        ego_trajectory = env.ego.unroll_trajectory(controls=torch.ones((10, 2)), dt=env.dt)
        ego_trajectory.requires_grad = True

        module = module_class(t_horizon=10, env=env)
        assert type(module.objective(ego_trajectory)) == float
        assert module.gradient(ego_trajectory, grad_wrt=ego_trajectory).size == ego_trajectory.numel()

    @staticmethod
    def test_runtime(module_class, env_class, num_modes):
        env = env_class(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1])})
        if num_modes > 1 and not env.is_multi_modal:
            pytest.skip()
        env.add_ado(position=torch.zeros(2), goal=torch.rand(2) * 10, num_modes=num_modes)
        env.add_ado(position=torch.tensor([5, 1]), goal=torch.rand(2) * (-10), num_modes=num_modes)
        ego_trajectory = env.ego.unroll_trajectory(controls=torch.ones((5, 2)) / 10.0, dt=env.dt)
        ego_trajectory.requires_grad = True

        module = module_class(t_horizon=5, env=env)
        objective_run_times, gradient_run_times = list(), list()
        for i in range(10):
            start_time = time.time()
            module.objective(ego_trajectory)
            objective_run_times.append(time.time() - start_time)

            start_time = time.time()
            module.gradient(ego_trajectory, grad_wrt=ego_trajectory)
            gradient_run_times.append(time.time() - start_time)

        assert np.mean(objective_run_times) < 0.03 * num_modes  # 33 Hz
        assert np.mean(gradient_run_times) < 0.05 * num_modes  # 20 Hz


def test_objective_goal_distribution():
    goal_state = torch.tensor([4.1, 8.9])
    ego_trajectory = torch.rand((11, 4))

    module = GoalModule(goal=goal_state, t_horizon=10, weight=1.0)
    module.importance_distribution = torch.zeros(module.importance_distribution.size())
    module.importance_distribution[3] = 1.0

    objective = module.objective(ego_trajectory)
    assert objective == torch.norm(ego_trajectory[3, 0:2] - goal_state)


###########################################################################
# Constraints #############################################################
###########################################################################
@pytest.mark.parametrize("module_class", CONSTRAINT_MODULES)
@pytest.mark.parametrize("env_class", ENVIRONMENTS)
@pytest.mark.parametrize("num_modes", [1, 2])
class TestConstraints:

    @staticmethod
    def test_runtime(module_class, env_class, num_modes):
        env = env_class(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1])})
        if num_modes > 1 and not env.is_multi_modal:
            pytest.skip()
        env.add_ado(position=torch.zeros(2), goal=torch.rand(2) * 10, num_modes=num_modes)
        env.add_ado(position=torch.tensor([5, 1]), goal=torch.rand(2) * (-10), num_modes=num_modes)
        ego_trajectory = env.ego.unroll_trajectory(controls=torch.ones((5, 2)) / 10.0, dt=env.dt)
        ego_trajectory.requires_grad = True

        module = module_class(env=env, t_horizon=5)
        constraint_run_times, jacobian_run_times = list(), list()
        for i in range(10):
            start_time = time.time()
            module.constraint(ego_trajectory)
            constraint_run_times.append(time.time() - start_time)

            start_time = time.time()
            module.jacobian(ego_trajectory, grad_wrt=ego_trajectory)
            jacobian_run_times.append(time.time() - start_time)

        assert np.mean(constraint_run_times) < 0.04 * num_modes  # 25 Hz
        assert np.mean(jacobian_run_times) < 0.05 * num_modes  # 20 Hz

    @staticmethod
    def test_violation(module_class, env_class, num_modes):
        """In order to test the constraint violation in general test it in a scene with static and far-distant
        agent(s), with  respect to the ego, and static ego robot. In this configurations all constraints should
        be met. """
        env = env_class(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1])})
        if num_modes > 1 and not env.is_multi_modal:
            pytest.skip()
        env.add_ado(position=torch.ones(2) * 9, goal=torch.ones(2) * 9, num_modes=num_modes)
        ego_trajectory = env.ego.unroll_trajectory(controls=torch.zeros((5, 2)), dt=env.dt)

        module = module_class(env=env, t_horizon=5)
        violation = module.compute_violation(ego_trajectory=ego_trajectory, ado_ids=None)
        assert violation == 0


@pytest.mark.parametrize("env_class", ENVIRONMENTS)
@pytest.mark.parametrize("num_modes", [1, 2])
def test_max_speed_constraint_violation(env_class, num_modes):
    env = env_class(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1]), "velocity": torch.zeros(2)})
    if num_modes > 1 and not env.is_multi_modal:
        pytest.skip()
    module = MaxSpeedModule(env=env, t_horizon=5)
    _, upper_bound = module.constraint_bounds

    # In this first scenario the ego has zero velocity over the full horizon.
    controls = torch.zeros((module._t_horizon, 2))
    ego_trajectory = env.ego.unroll_trajectory(controls=controls, dt=env.dt)
    violation = module.compute_violation(ego_trajectory=ego_trajectory)
    assert violation == 0

    # In this second scenario the ego has non-zero random velocity in x-direction, but always below the maximal
    # allowed speed (random -> [0, 1]).
    controls[:, 0] = torch.rand(module._t_horizon) * upper_bound  # single integrator, so no velocity summation !
    ego_trajectory = env.ego.unroll_trajectory(controls=controls, dt=env.dt)
    violation = module.compute_violation(ego_trajectory=ego_trajectory)
    assert violation == 0

    # In this third scenario the ego has the same random velocity in the x-direction as in the second scenario,
    # but at one time step, it is increased to a slightly larger speed than allowed.
    controls[1, 0] = upper_bound + 1e-3
    ego_trajectory = env.ego.unroll_trajectory(controls=controls, dt=env.dt)
    violation = module.compute_violation(ego_trajectory=ego_trajectory)
    assert violation > 0


@pytest.mark.parametrize("env_class", ENVIRONMENTS)
@pytest.mark.parametrize("num_modes", [1, 2])
def test_min_distance_constraint_violation(env_class, num_modes):
    env = env_class(IntegratorDTAgent, {"position": torch.ones(2) * 9, "velocity": torch.zeros(2)})
    if num_modes > 1 and not env.is_multi_modal:
        pytest.skip()
    ado_kwargs = {"goal": torch.tensor([9, -9]), "num_modes": num_modes}
    env.add_ado(position=torch.ones(2) * (-9), velocity=torch.tensor([1, 0]), **ado_kwargs)
    controls = torch.stack((torch.ones(10) * (-1), torch.zeros(10))).view(10, 2)
    ego_trajectory = env.ego.unroll_trajectory(controls=controls, dt=env.dt)

    # In this first scenario the ado and ego are moving parallel in maximal distance to each other.
    module = NormDistanceModule(env=env, t_horizon=controls.shape[0])
    lower_bound, _ = module.constraint_bounds
    violation = module.compute_violation(ego_trajectory=ego_trajectory)
    assert violation == 0

    # In the second scenario add another ado agent that is starting and moving very close to the ego robot.
    ado_start_pos = env.ego.position - (lower_bound * 0.5) * torch.ones(2)
    ado_kwargs = {"goal": ado_start_pos, "num_modes": num_modes}
    env.add_ado(position=ado_start_pos, velocity=torch.zeros(2), **ado_kwargs)
    module = NormDistanceModule(env=env, t_horizon=controls.shape[0])
    lower_bound, _ = module.constraint_bounds
    violation = module.compute_violation(ego_trajectory=ego_trajectory)
    assert violation > 0


###########################################################################
# Filter ##################################################################
###########################################################################
@pytest.mark.parametrize("module_class", FILTER_MODULES)
@pytest.mark.parametrize("env_class", ENVIRONMENTS)
@pytest.mark.parametrize("num_modes", [1, 2])
class TestFilter:

    @staticmethod
    def test_runtime(module_class, env_class, num_modes):
        env = env_class(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1])})
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
