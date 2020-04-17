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

        module = module_class(horizon=10, env=env)
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

        module = module_class(horizon=10, env=env)
        assert module.objective(ego_trajectory) is not None

    @staticmethod
    def test_output(module_class, env_class, num_modes):
        env = env_class(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1])})
        if num_modes > 1 and not env.is_multi_modal:
            pytest.skip()
        env.add_ado(position=torch.zeros(2), num_modes=num_modes)
        ego_trajectory = env.ego.unroll_trajectory(controls=torch.ones((10, 2)), dt=env.dt)
        ego_trajectory.requires_grad = True

        module = module_class(horizon=10, env=env)
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

        module = module_class(horizon=5, env=env)
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

    module = GoalModule(goal=goal_state, horizon=10, weight=1.0)
    module.importance_distribution = torch.zeros(module.importance_distribution.size())
    module.importance_distribution[3] = 1.0

    objective = module.objective(ego_trajectory)
    assert objective == torch.norm(ego_trajectory[3, 0:2] - goal_state)


###########################################################################
# Constraints #############################################################
###########################################################################
@pytest.mark.parametrize("module_class", CONSTRAINTS)
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

        module = module_class(env=env, horizon=10)
        constraint_run_times, jacobian_run_times = list(), list()
        for i in range(10):
            start_time = time.time()
            module.constraint(ego_trajectory)
            constraint_run_times.append(time.time() - start_time)

            start_time = time.time()
            module.jacobian(ego_trajectory, grad_wrt=ego_trajectory)
            jacobian_run_times.append(time.time() - start_time)

        assert np.mean(constraint_run_times) < 0.03 * num_modes  # 33 Hz
        assert np.mean(jacobian_run_times) < 0.04 * num_modes  # 25 Hz


###########################################################################
# Filter ##################################################################
###########################################################################
@pytest.mark.parametrize("module_class", FILTER)
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

        module = module_class()
        filter_run_times = list()
        for i in range(10):
            start_time = time.time()
            module.compute(*env.states())
            filter_run_times.append(time.time() - start_time)

        assert np.mean(filter_run_times) < 0.01  # 100 Hz
