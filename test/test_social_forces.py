import numpy as np

from mantrap.simulation import SocialForcesSimulation


def test_sf_single_ado_prediction():
    goal_position = np.array([2, 2])
    sim = SocialForcesSimulation()
    sim.add_ado(goal=goal_position, position=np.array([-1, -5]), velocity=np.ones(2) * 0.8)

    trajectory = np.squeeze(sim.predict(t_horizon=100))
    assert np.isclose(trajectory[-1][0], goal_position[0], atol=0.5)
    assert np.isclose(trajectory[-1][1], goal_position[1], atol=0.5)


def test_sf_static_ado_pair_prediction():
    sim = SocialForcesSimulation()
    sim.add_ado(goal=np.array([0, 0]), position=np.array([-1, 0]), velocity=np.array([0.1, 0]))
    sim.add_ado(goal=np.array([0, 0]), position=np.array([1, 0]), velocity=np.array([-0.1, 0]))

    trajectories = sim.predict(t_horizon=100)
    # Due to the repulsive of the agents between each other, they cannot both go to their goal position (which is
    # the same for both of them). Therefore the distance must be larger then zero basically, otherwise the repulsive
    # force would not act (or act attractive instead of repulsive).
    assert np.linalg.norm(trajectories[0, -1, 0:1] - trajectories[1, -1, 0:1]) > 1e-3
