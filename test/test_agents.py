import numpy as np
import pytest

from mantrap.agents import IntegratorDTAgent, DoubleIntegratorDTAgent


@pytest.mark.parametrize(
    "position, velocity, history, history_expected",
    [
        (np.zeros(2), np.zeros(2), None, np.zeros((1, 6))),
        (
            np.ones(2),
            np.zeros(2),
            np.reshape(np.array([5, 6, 0, 0, 0, -1]), (1, 6)),
            np.array([[5, 6, 0, 0, 0, -1], [1, 1, 0, 0, 0, 0]]),
        ),
    ],
)
def test_initialization(position: np.ndarray, velocity: np.ndarray, history: np.ndarray, history_expected: np.ndarray):
    agent = IntegratorDTAgent(position, velocity, history=history)
    assert np.array_equal(agent.history, history_expected)


@pytest.mark.parametrize(
    "position, velocity, velocity_input, dt, position_expected, velocity_expected",
    [
        (np.array([1, 0]), np.zeros(2), np.zeros(2), 1.0, np.array([1, 0]), np.zeros(2)),
        (np.array([1, 0]), np.array([2, 3]), np.zeros(2), 1.0, np.array([1, 0]), np.array([0, 0])),
        (np.array([1, 0]), np.zeros(2), np.array([2, 3]), 1.0, np.array([3, 3]), np.array([2, 3])),
    ],
)
def test_update_single_integrator(
    position: np.ndarray,
    velocity: np.ndarray,
    velocity_input: np.ndarray,
    dt: float,
    position_expected: np.ndarray,
    velocity_expected: np.ndarray,
):
    agent = IntegratorDTAgent(position, velocity)
    agent.update(velocity_input, dt=dt)
    assert np.array_equal(agent.position, position_expected)
    assert np.array_equal(agent.velocity, velocity_expected)


@pytest.mark.parametrize(
    "position, velocity, velocity_input, dt, position_expected, velocity_expected",
    [
        (np.array([1, 0]), np.zeros(2), np.zeros(2), 1.0, np.array([1, 0]), np.zeros(2)),
        (np.array([1, 0]), np.array([2, 3]), np.zeros(2), 1.0, np.array([3, 3]), np.array([2, 3])),
        (np.array([1, 0]), np.zeros(2), np.array([2, 3]), 1.0, np.array([2, 1.5]), np.array([2, 3])),
    ],
)
def test_update_double_integrator(
    position: np.ndarray,
    velocity: np.ndarray,
    velocity_input: np.ndarray,
    dt: float,
    position_expected: np.ndarray,
    velocity_expected: np.ndarray,
):
    agent = DoubleIntegratorDTAgent(position, velocity)
    agent.update(velocity_input, dt=dt)
    assert np.array_equal(agent.position, position_expected)
    assert np.array_equal(agent.velocity, velocity_expected)


def test_unrolling():
    ego = IntegratorDTAgent(np.zeros(2))
    policy = np.array([[1, 1], [2, 2], [4, 4]])
    trajectory = ego.unroll_trajectory(policy, dt=1.0)
    assert np.array_equal(trajectory[1:, 0:2], np.cumsum(policy, axis=0))


def test_reset():
    agent = IntegratorDTAgent(np.array([5, 6]))
    agent.reset(state=np.array([1, 5, 0.2, 4, 2, 1.0]), history=None)
    assert np.array_equal(agent.position, np.array([1, 5]))
    assert np.array_equal(agent.velocity, np.array([4, 2]))


@pytest.mark.parametrize("position, velocity, dt, n", [(np.array([-5, 0]), np.array([1, 0]), 1, 10)])
def test_ego_trajectory(position: np.ndarray, velocity: np.ndarray, dt: float, n: int):
    ego = IntegratorDTAgent(position=position, velocity=velocity)
    policy = np.vstack((np.ones(n) * velocity[0], np.ones(n) * velocity[1])).T
    ego_trajectory = ego.unroll_trajectory(policy=policy, dt=dt)

    assert np.array_equal(ego_trajectory[:, 0], np.linspace(position[0], position[0] + velocity[0] * n * dt, n + 1))
    assert np.array_equal(ego_trajectory[:, 1], np.linspace(position[1], position[1] + velocity[1] * n * dt, n + 1))
    assert np.array_equal(ego_trajectory[:, 2], np.zeros(n + 1))
    assert np.array_equal(ego_trajectory[:, 3], np.ones(n + 1) * velocity[0])
    assert np.array_equal(ego_trajectory[:, 4], np.ones(n + 1) * velocity[1])
    assert np.array_equal(ego_trajectory[:, 5], np.linspace(0, n, n + 1))
