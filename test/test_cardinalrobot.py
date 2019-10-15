import numpy as np

from murseco.robot import CardinalDTRobot
from murseco.utility.misc import cardinal_directions
import murseco.utility.io


def test_cardinalrobot_trajectory():
    position, thorizon, pstep = np.array([1.31, 4.3]), 10, 0.1
    policy = np.zeros((thorizon, 1))
    trajectory = CardinalDTRobot(position, thorizon, pstep, policy).trajectory()
    last_position = trajectory[-1, :]
    last_position_expected = position + cardinal_directions()[0, :] * thorizon * pstep
    assert np.isclose(np.linalg.norm(last_position - last_position_expected), 0)


def test_cardinalrobot_dynamics():
    position, velocity, direction = np.array([1.31, 4.3]), 0.1, 1
    robot = CardinalDTRobot(position, velocity=velocity)
    state_next = robot.dynamics(np.array([direction]))
    state_expected = position + cardinal_directions()[direction] * velocity
    assert np.isclose(np.linalg.norm(state_next - state_expected), 0)


def test_cardinalrobot_json():
    position, thorizon, pstep, policy = np.array([1.31, 4.3]), 10, 0.1, np.random.randint(0, 3, (10, 1))
    robot_1 = CardinalDTRobot(position, thorizon, pstep, policy)
    cache_path = murseco.utility.io.path_from_home_directory("test/cache/cardinalrobot_test.json")
    robot_1.to_json(cache_path)
    robot_2 = CardinalDTRobot.from_json(cache_path)
    assert robot_1.summary() == robot_2.summary()
