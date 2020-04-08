import time

import numpy as np
import pytest
import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.environment import PotentialFieldEnvironment, SocialForcesEnvironment
from mantrap.solver.constraints import MaxSpeedModule, MinDistanceModule
from mantrap.solver.constraints.constraint_module import ConstraintModule
from mantrap.solver.objectives import GoalModule, InteractionAccelerationModule, InteractionPositionModule
from mantrap.solver.objectives.objective_module import ObjectiveModule
from mantrap.utility.primitives import straight_line


###########################################################################
# Objectives ##############################################################
###########################################################################
@pytest.mark.parametrize("module_class", [InteractionAccelerationModule, InteractionPositionModule])
class TestObjectiveInteraction:

    @staticmethod
    def test_objective_far_and_near(module_class: ObjectiveModule.__class__):
        sim = PotentialFieldEnvironment(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1])})
        sim.add_ado(position=torch.zeros(2))
        x2_near = straight_line(torch.tensor([-5, 0.1]), torch.tensor([5, 0.1]), steps=11)
        x4_near = sim.ego.expand_trajectory(x2_near, dt=sim.dt)
        x2_far = straight_line(torch.tensor([-5, 10.0]), torch.tensor([5, 10.0]), steps=11)
        x4_far = sim.ego.expand_trajectory(x2_far, dt=sim.dt)

        module = module_class(horizon=10, env=sim)
        assert module.objective(x4_near) > module.objective(x4_far)

    @staticmethod
    def test_multimodal_support(module_class: ObjectiveModule.__class__):
        sim = SocialForcesEnvironment(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1])})
        ado_pos, ado_vel, ado_goal = torch.zeros(2), torch.tensor([-1, 0]), torch.tensor([-5, 0])
        sim.add_ado(position=ado_pos, velocity=ado_vel, num_modes=2, weights=[0.1, 0.9], goal=ado_goal)
        x4 = sim.ego.unroll_trajectory(controls=torch.ones((10, 2)), dt=sim.dt)

        module = module_class(horizon=10, env=sim)
        assert module.objective(x4) is not None

    @staticmethod
    def test_output(module_class: ObjectiveModule.__class__):
        sim = PotentialFieldEnvironment(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1])})
        sim.add_ado(position=torch.zeros(2))
        x4 = sim.ego.unroll_trajectory(controls=torch.ones((10, 2)), dt=sim.dt)
        x4.requires_grad = True

        module = module_class(horizon=10, env=sim)
        assert type(module.objective(x4)) == float
        assert module.gradient(x4, grad_wrt=x4).size == x4.numel()

    @staticmethod
    def test_runtime(module_class: ObjectiveModule.__class__):
        for sim_class in [PotentialFieldEnvironment, SocialForcesEnvironment]:
            sim = sim_class(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1])})
            sim.add_ado(position=torch.zeros(2), goal=torch.rand(2) * 10, num_modes=1)
            sim.add_ado(position=torch.tensor([5, 1]), goal=torch.rand(2) * (-10), num_modes=1)
            x4 = sim.ego.unroll_trajectory(controls=torch.ones((10, 2)) / 10.0, dt=sim.dt)
            x4.requires_grad = True

            module = module_class(horizon=10, env=sim)
            objective_run_times, gradient_run_times = list(), list()
            for i in range(10):
                start_time = time.time()
                module.objective(x4)
                objective_run_times.append(time.time() - start_time)

                start_time = time.time()
                module.gradient(x4, grad_wrt=x4)
                gradient_run_times.append(time.time() - start_time)

            assert np.mean(objective_run_times) < 0.03  # 33 Hz
            assert np.mean(gradient_run_times) < 0.05  # 20 Hz


def test_objective_goal_distribution():
    goal = torch.tensor([4.1, 8.9])
    x4 = torch.rand((11, 4))

    module = GoalModule(goal=goal, horizon=10, weight=1.0)
    module.importance_distribution = torch.zeros(module.importance_distribution.size())
    module.importance_distribution[3] = 1.0

    objective = module.objective(x4)
    assert objective == torch.norm(x4[3, 0:2] - goal)


###########################################################################
# Constraints #############################################################
###########################################################################
@pytest.mark.parametrize("module_class", [MaxSpeedModule, MinDistanceModule])
class TestConstraints:

    @staticmethod
    def test_runtime(module_class: ConstraintModule.__class__):
        for sim_class in [PotentialFieldEnvironment, SocialForcesEnvironment]:
            sim = sim_class(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1])})
            sim.add_ado(position=torch.zeros(2), goal=torch.rand(2) * 10, num_modes=1)
            sim.add_ado(position=torch.tensor([5, 1]), goal=torch.rand(2) * (-10), num_modes=1)
            x4 = sim.ego.unroll_trajectory(controls=torch.ones((9, 2)) / 10.0, dt=sim.dt)
            x4.requires_grad = True

            module = module_class(env=sim, horizon=10)
            constraint_run_times, jacobian_run_times = list(), list()
            for i in range(10):
                start_time = time.time()
                module.constraint(x4)
                constraint_run_times.append(time.time() - start_time)

                start_time = time.time()
                module.jacobian(x4, grad_wrt=x4)
                jacobian_run_times.append(time.time() - start_time)

            assert np.mean(constraint_run_times) < 0.02  # 50 Hz
            assert np.mean(jacobian_run_times) < 0.04  # 25 Hz
