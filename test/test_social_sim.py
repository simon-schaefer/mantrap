import numpy as np

from mantrap.agents import DoubleIntegratorDTAgent, IntegratorDTAgent
import mantrap.simulation
from mantrap.utility.io import path_from_home_directory
from mantrap.visualization import plot_scene


def test_ego_initialization():
    ego_position = np.array([4, 6])
    sim = mantrap.simulation.SocialForcesSimulation(IntegratorDTAgent, {"position": ego_position}, dt=1.0)
    assert np.array_equal(sim.ego.position, ego_position)


def test_single_ado_prediction():
    goal_position = np.array([2, 2])
    sim = mantrap.simulation.SocialForcesSimulation()
    sim.add_ado(goal_position=goal_position, position=np.array([-1, -5]), velocity=np.ones(2) * 0.8)

    trajectory = sim.predict(t_horizon=100)[0, :, :]
    assert np.isclose(trajectory[-1, 0], goal_position[0], atol=0.5)
    assert np.isclose(trajectory[-1, 1], goal_position[1], atol=0.5)


def test_static_ado_pair_prediction():
    sim = mantrap.simulation.SocialForcesSimulation()
    sim.add_ado(goal_position=np.array([0, 0]), position=np.array([-1, 0]), velocity=np.array([0.1, 0]))
    sim.add_ado(goal_position=np.array([0, 0]), position=np.array([1, 0]), velocity=np.array([-0.1, 0]))

    trajectories = sim.predict(t_horizon=100)
    # Due to the repulsive of the agents between each other, they cannot both go to their goal position (which is
    # the same for both of them). Therefore the distance must be larger then zero basically, otherwise the repulsive
    # force would not act (or act attractive instead of repulsive).
    assert np.linalg.norm(trajectories[0, -1, 0:1] - trajectories[1, -1, 0:1]) > 1e-3


def visualize_ados_cross_prediction():
    sim = mantrap.simulation.SocialForcesSimulation()
    sim.add_ado(goal_position=np.array([5, 5]), position=np.array([-5, -5]), velocity=np.array([1, 1]))
    sim.add_ado(goal_position=np.array([-5, -5]), position=np.array([5, 5]), velocity=np.array([-1, -1]))
    sim.add_ado(goal_position=np.array([-5, 5]), position=np.array([5, -5]), velocity=np.array([-1, 1]))
    sim.add_ado(goal_position=np.array([5, -5]), position=np.array([-5, 5]), velocity=np.array([1, -1]))
    output_dir = path_from_home_directory("test/graphs/social_sim_ados_cross")

    for t in range(100):
        ado_trajectories = sim.step()
        plot_scene(sim, ado_trajectories=ado_trajectories, output_dir=output_dir, image_tag=f"{sim.sim_time:.2f}")
    

def visualize_dodging_prediction():
    sim = mantrap.simulation.SocialForcesSimulation(x_axis=(-2, 2), y_axis=(-2, 2))
    sim.add_ado(goal_position=np.array([0, -1.5]), position=np.array([0, 0]), velocity=np.array([0, -0.1]))
    sim.add_ado(goal_position=np.array([1.5, 0]), position=np.array([-1.5, 0]), velocity=np.array([0.2, 0.4]))
    output_dir = path_from_home_directory("test/graphs/social_sim_dodging")

    for t in range(100):
        ado_trajectories = sim.step()
        plot_scene(sim, ado_trajectories=ado_trajectories, output_dir=output_dir, image_tag=f"{sim.sim_time:.2f}")
