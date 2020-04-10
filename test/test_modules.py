import time

import numpy as np
import pytest
import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.environment import PotentialFieldEnvironment, SocialForcesEnvironment, Trajectron
from mantrap.environment.environment import GraphBasedEnvironment
from mantrap.solver.constraints import MaxSpeedModule, MinDistanceModule
from mantrap.solver.constraints.constraint_module import ConstraintModule
from mantrap.solver.filter import EuclideanModule, NoFilterModule
from mantrap.solver.filter.filter_module import FilterModule
from mantrap.solver.objectives import GoalModule, InteractionAccelerationModule, InteractionPositionModule
from mantrap.solver.objectives.objective_module import ObjectiveModule
from mantrap.utility.primitives import straight_line


###########################################################################
# Objectives ##############################################################
###########################################################################
@pytest.mark.parametrize(
    "module_class, env_class", [
        (InteractionPositionModule, PotentialFieldEnvironment),
        (InteractionPositionModule, SocialForcesEnvironment),
        (InteractionPositionModule, Trajectron),
        #(InteractionAccelerationModule, PotentialFieldEnvironment),
        #(InteractionAccelerationModule, SocialForcesEnvironment),
        #(InteractionAccelerationModule, Trajectron)
    ]
)
class TestObjectiveInteraction:

    @staticmethod
    def test_far_and_near(module_class: ObjectiveModule.__class__, env_class: GraphBasedEnvironment.__class__):
        """Every interaction-based objective should be larger the closer the interacting agents are, so having the
        ego agent close to some ado should affect the ado more than when the ego agent is far away. """
        env = env_class(IntegratorDTAgent, {"position": torch.tensor([-5, 100.0])}, y_axis=(-100, 100))
        env.add_ado(position=torch.zeros(2))

        ego_path_near = straight_line(torch.tensor([-5, 0.1]), torch.tensor([5, 0.1]), steps=11)
        ego_trajectory_near = env.ego.expand_trajectory(ego_path_near, dt=env.dt)
        ego_path_far = straight_line(torch.tensor([-5, 100.0]), torch.tensor([5, 10.0]), steps=11)
        ego_trajectory_far = env.ego.expand_trajectory(ego_path_far, dt=env.dt)

        module = module_class(horizon=10, env=env)
        assert module.objective(ego_trajectory_near) > module.objective(ego_trajectory_far)

    @staticmethod
    def test_multimodal_support(module_class: ObjectiveModule.__class__, env_class: GraphBasedEnvironment.__class__):
        env = env_class(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1])})
        ado_pos, ado_vel, ado_goal = torch.zeros(2), torch.tensor([-1, 0]), torch.tensor([-5, 0])
        env.add_ado(position=ado_pos, velocity=ado_vel, num_modes=2, weights=[0.1, 0.9], goal=ado_goal)
        ego_trajectory = env.ego.unroll_trajectory(controls=torch.ones((10, 2)), dt=env.dt)

        module = module_class(horizon=10, env=env)
        assert module.objective(ego_trajectory) is not None

    @staticmethod
    def test_output(module_class: ObjectiveModule.__class__, env_class: GraphBasedEnvironment.__class__):
        env = env_class(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1])})
        env.add_ado(position=torch.zeros(2))
        ego_trajectory = env.ego.unroll_trajectory(controls=torch.ones((10, 2)), dt=env.dt)
        ego_trajectory.requires_grad = True

        module = module_class(horizon=10, env=env)
        assert type(module.objective(ego_trajectory)) == float
        assert module.gradient(ego_trajectory, grad_wrt=ego_trajectory).size == ego_trajectory.numel()

    @staticmethod
    def test_runtime(module_class: ObjectiveModule.__class__, env_class: GraphBasedEnvironment.__class__):
        env = env_class(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1])})
        env.add_ado(position=torch.zeros(2), goal=torch.rand(2) * 10, num_modes=1)
        env.add_ado(position=torch.tensor([5, 1]), goal=torch.rand(2) * (-10), num_modes=1)
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

        assert np.mean(objective_run_times) < 0.03  # 33 Hz
        assert np.mean(gradient_run_times) < 0.05  # 20 Hz


def test_objective_goal_distribution():
    goal = torch.tensor([4.1, 8.9])
    ego_trajectory = torch.rand((11, 4))

    module = GoalModule(goal=goal, horizon=10, weight=1.0)
    module.importance_distribution = torch.zeros(module.importance_distribution.size())
    module.importance_distribution[3] = 1.0

    objective = module.objective(ego_trajectory)
    assert objective == torch.norm(ego_trajectory[3, 0:2] - goal)


###########################################################################
# Constraints #############################################################
###########################################################################
@pytest.mark.parametrize(
    "module_class, env_class", [
        (MaxSpeedModule, PotentialFieldEnvironment),
        (MaxSpeedModule, SocialForcesEnvironment),
        #(MaxSpeedModule, Trajectron),
        (MinDistanceModule, PotentialFieldEnvironment),
        (MinDistanceModule, SocialForcesEnvironment),
        #(MinDistanceModule, Trajectron)
    ]
)
class TestConstraints:

    @staticmethod
    def test_runtime(module_class: ConstraintModule.__class__, env_class: GraphBasedEnvironment.__class__):
        env = env_class(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1])})
        env.add_ado(position=torch.zeros(2), goal=torch.rand(2) * 10, num_modes=1)
        env.add_ado(position=torch.tensor([5, 1]), goal=torch.rand(2) * (-10), num_modes=1)
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

        assert np.mean(constraint_run_times) < 0.03  # 33 Hz
        assert np.mean(jacobian_run_times) < 0.04  # 25 Hz


###########################################################################
# Filter ##################################################################
###########################################################################
@pytest.mark.parametrize(
    "module_class, env_class", [
        (EuclideanModule, PotentialFieldEnvironment),
        (EuclideanModule, SocialForcesEnvironment),
        (EuclideanModule, Trajectron),
        (NoFilterModule, PotentialFieldEnvironment),
        (NoFilterModule, SocialForcesEnvironment),
        (NoFilterModule, Trajectron)
    ]
)
class TestFilter:

    @staticmethod
    def test_runtime(module_class: FilterModule.__class__, env_class: GraphBasedEnvironment.__class__):
        env = env_class(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1])})
        env.add_ado(position=torch.zeros(2), goal=torch.rand(2) * 10, num_modes=2)
        env.add_ado(position=torch.tensor([5, 1]), goal=torch.rand(2) * (-10), num_modes=2)

        module = module_class()
        filter_run_times = list()
        for i in range(10):
            start_time = time.time()
            module.compute(*env.states())
            filter_run_times.append(time.time() - start_time)

        assert np.mean(filter_run_times) < 0.01  # 100 Hz
