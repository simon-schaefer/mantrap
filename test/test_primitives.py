import pytest
import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.constants import agent_speed_max
from mantrap.utility.primitives import square_primitives


@pytest.mark.parametrize("num_points", [5, 10])
def test_square_primitives(num_points: int):
    position, velocity, goal = torch.tensor([-5, 0]), torch.tensor([1, 0]), torch.tensor([2, 0])
    agent = IntegratorDTAgent(position=position, velocity=velocity)
    primitives = square_primitives(start=agent.position, end=goal, dt=1.0, num_points=num_points)
    print(primitives.shape)

    assert primitives.shape[1] == num_points
    for m in range(primitives.shape[0]):
        for i in range(1, num_points - 1):
            distance = torch.norm(primitives[m, i, :] - primitives[m, i - 1, :])
            distance_next = torch.norm(primitives[m, i + 1, :] - primitives[m, i, :])
            if torch.isclose(distance_next.float(), torch.zeros(1), atol=0.1):
                continue
            assert torch.isclose(distance.float(), torch.tensor([agent_speed_max]).float(), atol=0.5)  # dt = 1.0

    # The center primitive should be a straight line, therefore the one with largest x-expansion, since we are moving
    # straight in x-direction. Similarly the first primitive should have the largest expansion in y direction, the
    # last one the smallest.
    assert all([primitives[1, -1, 0] >= primitives[i, -1, 0] for i in range(primitives.shape[0])])
    assert all([primitives[0, -1, 1] >= primitives[i, -1, 1] for i in range(primitives.shape[0])])
    assert all([primitives[-1, -1, 1] <= primitives[i, -1, 1] for i in range(primitives.shape[0])])
