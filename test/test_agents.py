import numpy as np
import pytest

import mantrap.agents


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
    agent = mantrap.agents.IntegratorDTAgent(position, velocity, history=history)
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
    agent = mantrap.agents.IntegratorDTAgent(position, velocity)
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
    agent = mantrap.agents.DoubleIntegratorDTAgent(position, velocity)
    agent.update(velocity_input, dt=dt)
    assert np.array_equal(agent.position, position_expected)
    assert np.array_equal(agent.velocity, velocity_expected)


def test_unrolling():
    ego = mantrap.agents.IntegratorDTAgent(np.zeros(2))
    policy = np.array([[1, 1], [2, 2], [4, 4]])
    trajectory = ego.unroll_trajectory(policy, dt=1.0)
    assert np.array_equal(trajectory[1:, 0:2], np.cumsum(policy, axis=0))
