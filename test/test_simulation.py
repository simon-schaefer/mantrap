import numpy as np

from mantrap.agents import IntegratorDTAgent
from mantrap.simulation import DistanceFieldSimulation, SocialForcesSimulation
from mantrap.utility.io import path_from_home_directory
from mantrap.visualization import plot_scene


###########################################################################
# Abstract simulation test ################################################
# The abstract simulation itself is an abstract class and can therefore not
# be initialized. Therefore the distance field class is used for testing
# the abstract simulation since it has very little overhead and is pretty
# simple, so that the units of the abstract class can be tested without
# much (possible) "noise" from other overhead.
###########################################################################
def test_initialization():
    ego_position = np.array([4, 6])
    sim = DistanceFieldSimulation(IntegratorDTAgent, {"position": ego_position}, dt=1.0)
    assert np.array_equal(sim.ego.position, ego_position)
    assert sim.num_ados == 0
    assert sim.sim_time == 0.0


def test_step():
    ado_position = np.zeros(2)
    ego_position = np.array([-4, 6])
    sim = DistanceFieldSimulation(IntegratorDTAgent, {"position": ego_position}, dt=1.0)
    sim.add_ado(position=ado_position)
    assert sim.num_ados == 1

    ego_policy = np.array([1, 0])
    num_steps = 10
    ado_trajectory = np.zeros((sim.num_ados, 1, num_steps, 6))
    ego_trajectory = np.zeros((num_steps, 6))
    for t in range(num_steps):
        ado_t, ego_t = sim.step(ego_policy=ego_policy)
        assert ado_t.size == 6
        assert ado_t.shape == (1, 1, 6)
        assert ego_t.size == 6
        ado_trajectory[:, :, t, :] = ado_t
        ego_trajectory[t, :] = ego_t

    ego_t_exp = np.vstack(
        (
            np.linspace(ego_position[0], ego_position[0] + ego_policy[0] * sim.dt * num_steps, num_steps + 1),
            np.linspace(ego_position[1], ego_position[1] + ego_policy[1] * sim.dt * num_steps, num_steps + 1),
            np.zeros(num_steps + 1),
            np.ones(num_steps + 1) * ego_policy[0],
            np.ones(num_steps + 1) * ego_policy[1],
            np.linspace(0, num_steps * sim.dt, num_steps + 1),
        )
    ).T
    assert np.array_equal(ego_trajectory, ego_t_exp[1:, :])


def test_update():
    sim = DistanceFieldSimulation(ego_type=IntegratorDTAgent, ego_kwargs={"position": np.zeros(2)}, dt=1)

    num_steps = 10
    sim_times = np.zeros(num_steps)
    ego_positions = np.zeros((num_steps, 2))
    for t in range(num_steps):
        sim_times[t] = sim.sim_time
        ego_positions[t, :] = sim.ego.position
        sim.step(ego_policy=np.array([1, 0]))
    assert np.array_equal(sim_times, np.linspace(0, num_steps - 1, num_steps))
    assert np.array_equal(ego_positions[:, 0], np.linspace(0, num_steps - 1, num_steps))
    assert np.array_equal(ego_positions[:, 1], np.zeros(num_steps))


###########################################################################
# Distance Field simulation specific tests ################################
###########################################################################
def test_df_graph():
    sim = DistanceFieldSimulation()
    graph = sim.build_graph()
    assert sim.graph_check(graph=graph)


def test_df_potential_field():
    sim = DistanceFieldSimulation(x_axis=(-2, 2), y_axis=(-2, 2))
    sim.add_ado(position=np.zeros(2), velocity=np.zeros(2))

    num_grid_points = 31
    assert num_grid_points % 2 == 1  # number must be odd for tests below
    forces_norm_grid = np.zeros((num_grid_points, num_grid_points))
    for ix, x in enumerate(np.linspace(-2, 2, num_grid_points)):
        for iy, y in enumerate(np.linspace(-2, 2, num_grid_points)):
            graph = sim.build_graph(ego_state=np.array([x, y]))
            forces_norm_grid[ix, iy] = np.linalg.norm(graph["forces_sum"].detach().numpy())

    # Test full symmetry and maximum in the middle.
    ix_max, iy_max = np.unravel_index(np.argmax(forces_norm_grid), shape=forces_norm_grid.shape)
    assert ix_max == num_grid_points // 2  # shift by one index due to np.lin-space-operator
    assert iy_max == num_grid_points // 2  # shift by one index due to np.lin-space-operator

    assert np.isclose(np.linalg.norm(forces_norm_grid - np.flip(forces_norm_grid, axis=0)), 0.0)
    assert np.isclose(np.linalg.norm(forces_norm_grid - np.flip(forces_norm_grid, axis=1)), 0.0)


###########################################################################
# Social Forces simulation specific tests #################################
###########################################################################
def test_sf_single_ado_prediction():
    goal_position = np.array([2, 2])
    sim = SocialForcesSimulation()
    sim.add_ado(goal_position=goal_position, position=np.array([-1, -5]), velocity=np.ones(2) * 0.8)

    trajectory = sim.predict(t_horizon=100)[0, :, :]
    assert np.isclose(trajectory[-1][0], goal_position[0], atol=0.5)
    assert np.isclose(trajectory[-1][1], goal_position[1], atol=0.5)


def test_sf_static_ado_pair_prediction():
    sim = SocialForcesSimulation()
    sim.add_ado(goal_position=np.array([0, 0]), position=np.array([-1, 0]), velocity=np.array([0.1, 0]))
    sim.add_ado(goal_position=np.array([0, 0]), position=np.array([1, 0]), velocity=np.array([-0.1, 0]))

    trajectories = sim.predict(t_horizon=100)
    # Due to the repulsive of the agents between each other, they cannot both go to their goal position (which is
    # the same for both of them). Therefore the distance must be larger then zero basically, otherwise the repulsive
    # force would not act (or act attractive instead of repulsive).
    assert np.linalg.norm(trajectories[0, -1, 0:1] - trajectories[1, -1, 0:1]) > 1e-3


###########################################################################
# Visualization ###########################################################
###########################################################################
def visualize_sf_ados_cross_prediction():
    sim = SocialForcesSimulation()
    sim.add_ado(goal_position=np.array([4, 5]), position=np.array([-5, -5]), velocity=np.array([1, 1]))
    sim.add_ado(goal_position=np.array([-5, -4]), position=np.array([5, 5]), velocity=np.array([-1, -1]))
    sim.add_ado(goal_position=np.array([-3, 5]), position=np.array([5, -5]), velocity=np.array([-1, 1]))
    sim.add_ado(goal_position=np.array([5, -2]), position=np.array([-5, 5]), velocity=np.array([1, -1]))
    output_dir = path_from_home_directory("test/graphs/social_sim_ados_cross")

    num_steps = 100
    ado_trajectories = np.zeros((sim.num_ados, 1, num_steps, 6))
    for t in range(num_steps):
        ado_trajectories[:, :, t, :], _ = sim.step()
    plot_scene(ado_trajectories, ado_colors=sim.ado_colors, output_dir=output_dir)


def visualize_sf_agent_and_ego_prediction():
    sim = SocialForcesSimulation(
        ego_type=IntegratorDTAgent,
        ego_kwargs={"position": np.array([-5, 0]), "velocity": np.array([1, 0])},
        fluctuations=0.0,
        x_axis=(-5, 5),
        y_axis=(-5, 5),
    )
    sim.add_ado(goal_position=np.array([0, 0]), position=np.array([0, 0]), velocity=np.array([0, 0]))
    ego_policy = np.vstack((np.ones(50), np.zeros(50))).T
    output_dir = path_from_home_directory("test/graphs/social_sim_ego_dodging")

    num_steps = 50
    ado_trajectories = np.zeros((sim.num_ados, 1, num_steps, 6))
    ego_trajectory = np.zeros((num_steps, 6))
    for t in range(num_steps):
        ado_trajectories[:, :, t, :], ego_trajectory[t, :] = sim.step(ego_policy=ego_policy)
    plot_scene(ado_trajectories, ado_colors=sim.ado_colors, ego_trajectory=ego_trajectory, output_dir=output_dir)
