import time

import numpy as np
import pytest
import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.simulation import PotentialFieldSimulation, SocialForcesSimulation
from mantrap.solver.objectives import GoalModule, InteractionAccelerationModule, InteractionPositionModule
from mantrap.solver.objectives.objective_module import ObjectiveModule
from mantrap.utility.primitives import straight_line


@pytest.mark.parametrize("module_class", [InteractionAccelerationModule, InteractionPositionModule])
class TestObjectiveInteraction:

    @staticmethod
    def test_objective_far_and_near(module_class: ObjectiveModule.__class__):
        sim = PotentialFieldSimulation(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1])})
        sim.add_ado(position=torch.zeros(2))
        x2_near = straight_line(torch.tensor([-5, 0.1]), torch.tensor([5, 0.1]), 10)
        x2_far = straight_line(torch.tensor([-5, 10.0]), torch.tensor([5, 10.0]), 10)

        module = module_class(horizon=10, env=sim)
        assert module.objective(x2_near) > module.objective(x2_far)

    @staticmethod
    def test_multimodal_support(module_class: ObjectiveModule.__class__):
        sim = SocialForcesSimulation(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1])})
        ado_pos, ado_vel, ado_goal = torch.zeros(2), torch.tensor([-1, 0]), torch.tensor([-5, 0])
        sim.add_ado(position=ado_pos, velocity=ado_vel, num_modes=2, weights=[0.1, 0.9], goal=ado_goal)
        x2 = straight_line(sim.ego.position, torch.tensor([5, 0.1]), 10)

        module = module_class(horizon=10, env=sim)
        assert module.objective(x2) is not None

    @staticmethod
    def test_output(module_class: ObjectiveModule.__class__):
        sim = PotentialFieldSimulation(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1])})
        sim.add_ado(position=torch.zeros(2))
        x2 = straight_line(sim.ego.position, torch.tensor([5, 0.1]), 10)

        module = module_class(horizon=10, env=sim)
        assert type(module.objective(x2)) == float
        assert module.gradient(x2).size == x2.numel()  # without setting `grad_wrt`, size should match x2

    @staticmethod
    def test_runtime(module_class: ObjectiveModule.__class__):
        for sim_class in [PotentialFieldSimulation, SocialForcesSimulation]:
            sim = sim_class(IntegratorDTAgent, {"position": torch.tensor([-5, 0.1])})
            sim.add_ado(position=torch.zeros(2), goal=torch.rand(2) * 10, num_modes=1)
            sim.add_ado(position=torch.tensor([5, 1]), goal=torch.rand(2) * (-10), num_modes=1)
            x2 = straight_line(sim.ego.position, torch.tensor([5, 0.1]), 10)

            module = module_class(horizon=10, env=sim)
            objective_run_times, gradient_run_times = list(), list()
            for i in range(10):
                start_time = time.time()
                module.objective(x2)
                objective_run_times.append(time.time() - start_time)

                start_time = time.time()
                module.gradient(x2)
                gradient_run_times.append(time.time() - start_time)

            assert np.mean(objective_run_times) < 0.03  # 33 Hz
            assert np.mean(gradient_run_times) < 0.05  # 20 Hz


def test_objective_goal_distribution():
    goal = torch.tensor([4.1, 8.9])
    x2 = torch.rand((10, 2))

    module = GoalModule(goal=goal, horizon=10, weight=1.0)
    module.importance_distribution = torch.zeros(10)
    module.importance_distribution[3] = 1.0

    objective = module.objective(x2)
    assert objective == torch.norm(x2[3, :] - goal)
