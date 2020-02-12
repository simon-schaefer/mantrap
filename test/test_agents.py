import pytest
import torch

from mantrap.agents import IntegratorDTAgent, DoubleIntegratorDTAgent


@pytest.mark.parametrize(
    "pos, vel, history, history_expected",
    [
        (torch.zeros(2), torch.zeros(2), None, torch.zeros((1, 5))),
        (
            torch.ones(2),
            torch.zeros(2),
            torch.tensor([5, 6, 0, 0, -1]).view(1, 5),
            torch.tensor([[5, 6, 0, 0, -1], [1, 1, 0, 0, 0]]),
        ),
    ],
)
def test_initialization(pos: torch.Tensor, vel: torch.Tensor, history: torch.Tensor, history_expected: torch.Tensor):
    agent = IntegratorDTAgent(position=pos, velocity=vel, history=history)
    assert torch.all(torch.eq(agent.history, history_expected))


@pytest.mark.parametrize(
    "position, velocity, velocity_input, dt, position_expected, velocity_expected",
    [
        (torch.tensor([1, 0]), torch.zeros(2), torch.zeros(2), 1.0, torch.tensor([1, 0]), torch.zeros(2)),
        (torch.tensor([1, 0]), torch.tensor([2, 3]), torch.zeros(2), 1.0, torch.tensor([1, 0]), torch.tensor([0, 0])),
        (torch.tensor([1, 0]), torch.zeros(2), torch.tensor([2, 3]), 1.0, torch.tensor([3, 3]), torch.tensor([2, 3])),
    ],
)
def test_update_single_integrator(
    position: torch.Tensor,
    velocity: torch.Tensor,
    velocity_input: torch.Tensor,
    dt: float,
    position_expected: torch.Tensor,
    velocity_expected: torch.Tensor,
):
    agent = IntegratorDTAgent(position, velocity)
    agent.update(velocity_input, dt=dt)
    assert torch.all(torch.eq(agent.position, position_expected))
    assert torch.all(torch.eq(agent.velocity, velocity_expected))


@pytest.mark.parametrize(
    "position, velocity, velocity_input, dt, position_expected, velocity_expected",
    [
        (torch.tensor([1, 0]), torch.zeros(2), torch.zeros(2), 1.0, torch.tensor([1, 0]), torch.zeros(2)),
        (torch.tensor([1, 0]), torch.tensor([2, 3]), torch.zeros(2), 1.0, torch.tensor([3, 3]), torch.tensor([2, 3])),
        (torch.tensor([1, 0]), torch.zeros(2), torch.tensor([2, 3]), 1.0, torch.tensor([2, 1.5]), torch.tensor([2, 3])),
    ],
)
def test_update_double_integrator(
    position: torch.Tensor,
    velocity: torch.Tensor,
    velocity_input: torch.Tensor,
    dt: float,
    position_expected: torch.Tensor,
    velocity_expected: torch.Tensor,
):
    agent = DoubleIntegratorDTAgent(position, velocity)
    agent.update(velocity_input, dt=dt)
    assert torch.all(torch.eq(agent.position, position_expected))
    assert torch.all(torch.eq(agent.velocity, velocity_expected))


def test_unrolling():
    ego = IntegratorDTAgent(torch.zeros(2))
    policy = torch.tensor([[1, 1], [2, 2], [4, 4]])
    trajectory = ego.unroll_trajectory(policy, dt=1.0)
    assert torch.all(torch.eq(trajectory[1:, 0:2], torch.cumsum(policy, dim=0)))


def test_reset():
    agent = IntegratorDTAgent(torch.tensor([5, 6]))
    agent.reset(state=torch.tensor([1, 5, 4, 2, 1.0]), history=None)
    assert torch.all(torch.eq(agent.position, torch.tensor([1, 5])))
    assert torch.all(torch.eq(agent.velocity, torch.tensor([4, 2])))


@pytest.mark.parametrize("position, velocity, dt, n", [(torch.tensor([-5, 0]), torch.tensor([1, 0]), 1, 10)])
def test_ego_trajectory(position: torch.Tensor, velocity: torch.Tensor, dt: float, n: int):
    ego = IntegratorDTAgent(position=position, velocity=velocity)
    policy = torch.stack((torch.ones(n) * velocity[0], torch.ones(n) * velocity[1])).T
    ego_trajectory = ego.unroll_trajectory(policy=policy, dt=dt)

    ego_trajectory_x_exp = torch.linspace(position[0].item(), position[0].item() + velocity[0].item() * n * dt, n + 1)
    ego_trajectory_y_exp = torch.linspace(position[1].item(), position[1].item() + velocity[1].item() * n * dt, n + 1)
    assert torch.all(torch.eq(ego_trajectory[:, 0], ego_trajectory_x_exp))
    assert torch.all(torch.eq(ego_trajectory[:, 1], ego_trajectory_y_exp))
    assert torch.all(torch.eq(ego_trajectory[:, 2], torch.ones(n + 1) * velocity[0]))
    assert torch.all(torch.eq(ego_trajectory[:, 3], torch.ones(n + 1) * velocity[1]))
    assert torch.all(torch.eq(ego_trajectory[:, 4], torch.linspace(0, n, n + 1)))


def test_history():
    agent = IntegratorDTAgent(position=torch.tensor([-1, 4]), velocity=torch.ones(2))
    for _ in range(4):
        agent.update(action=torch.ones(2))
    assert len(agent.history.shape) == 2
    assert agent.history.shape[0] == 5
    assert agent.history.shape[1] == 5
