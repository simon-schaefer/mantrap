import pytest
import torch

import mantrap.agents
import mantrap.constants
import mantrap.environment
import mantrap.utility.maths
import mantrap.utility.shaping


###########################################################################
# Tests - All Environment #################################################
###########################################################################
@pytest.mark.parametrize("environment_class", [mantrap.environment.KalmanEnvironment,
                                               mantrap.environment.PotentialFieldEnvironment,
                                               mantrap.environment.SocialForcesEnvironment,
                                               mantrap.environment.Trajectron])
class TestEnvironment:

    @staticmethod
    def test_initialization(environment_class: mantrap.environment.base.GraphBasedEnvironment.__class__):
        ego_position = torch.rand(2).float()
        env = environment_class(ego_type=mantrap.agents.IntegratorDTAgent, ego_position=ego_position)
        assert torch.all(torch.eq(env.ego.position, ego_position))
        assert env.num_ados == 0
        assert env.time == 0.0
        env.add_ado(position=torch.tensor([6, 7]), velocity=torch.ones(2))

        assert torch.all(torch.eq(env.ados[0].position, torch.tensor([6, 7]).float()))
        assert torch.all(torch.eq(env.ados[0].velocity, torch.ones(2)))

    @staticmethod
    def test_step(environment_class: mantrap.environment.base.GraphBasedEnvironment.__class__):
        ado_init_position = torch.zeros(2)
        ado_init_velocity = torch.ones(2)
        ego_init_position = torch.tensor([-4, 6])
        env = environment_class(ego_type=mantrap.agents.IntegratorDTAgent, ego_position=ego_init_position)

        # In order to be able to verify the generated trajectories easily, we assume uni-modality here.
        env.add_ado(position=ado_init_position, velocity=ado_init_velocity)
        assert env.num_ados == 1

        t_horizon = 5
        ego_controls = torch.stack([torch.tensor([1, 0])] * t_horizon)
        ego_trajectory = env.ego.unroll_trajectory(controls=ego_controls, dt=env.dt)
        for t in range(t_horizon):
            ado_t, ego_t = env.step(ego_action=ego_controls[t])

            # Check dimensions of outputted ado and ego states.
            assert ado_t.numel() == 5
            assert ado_t.shape == (1, 5)
            assert ego_t.numel() == 5

            # While the exact value of the ado agent's states depends on the environment dynamics used, all of them
            # are based on the ego state (control), which is thought to be enforced while forwarding the environment.
            assert all(torch.isclose(ego_t, ego_trajectory[t+1, :]))

    @staticmethod
    def test_step_reset(environment_class: mantrap.environment.base.GraphBasedEnvironment.__class__):
        ego_position = torch.rand(2)
        env = environment_class(ego_type=mantrap.agents.IntegratorDTAgent, ego_position=ego_position)

        # In order to be able to verify the generated trajectories easily, we assume uni-modality here.
        env.add_ado(position=torch.zeros(2), velocity=torch.zeros(2))
        env.add_ado(position=torch.ones(2), velocity=torch.zeros(2))

        ego_next_state = torch.rand(5)
        ado_next_states = torch.rand(env.num_ados, 5)
        env.step_reset(ego_next=ego_next_state, ado_next=ado_next_states)

        assert torch.all(torch.eq(env.ego.state_with_time, ego_next_state))
        for m_ado, ado in enumerate(env.ados):
            assert torch.allclose(ado.state_with_time, ado_next_states[m_ado, :])

    @staticmethod
    def test_prediction_trajectories_shape(environment_class: mantrap.environment.base.GraphBasedEnvironment.__class__):
        env = environment_class()

        t_horizon = 4
        history = torch.stack(5 * [torch.tensor([1, 0, 0, 0, 0])])
        env.add_ado(goal=torch.ones(2), position=torch.tensor([-1, 0]), history=history)
        env.add_ado(goal=torch.zeros(2), position=torch.tensor([1, 0]), history=history)

        ado_trajectories = env.sample_wo_ego(t_horizon=t_horizon)
        assert mantrap.utility.shaping.check_ado_samples(ado_trajectories, t_horizon=t_horizon + 1, ados=2)

    @staticmethod
    def test_build_distributions(environment_class: mantrap.environment.base.GraphBasedEnvironment.__class__):
        ego_position = torch.rand(2)
        env = environment_class(ego_type=mantrap.agents.IntegratorDTAgent, ego_position=ego_position)
        env.add_ado(position=torch.tensor([3, 0]), goal=torch.tensor([-4, 0]))
        env.add_ado(position=torch.tensor([5, 0]), goal=torch.tensor([-2, 0]))
        env.add_ado(position=torch.tensor([10, 0]), goal=torch.tensor([5, 3]))

        prediction_horizon = 10
        trajectory = torch.zeros((prediction_horizon + 1, 4))  # does not matter here anyway
        dist_dict = env.compute_distributions(ego_trajectory=trajectory)

        assert env.check_distribution(dist_dict, t_horizon=prediction_horizon)

    @staticmethod
    def test_detaching(environment_class: mantrap.environment.base.GraphBasedEnvironment.__class__):
        ego_position = torch.rand(2)
        env = environment_class(ego_type=mantrap.agents.IntegratorDTAgent, ego_position=ego_position)
        env.add_ado(position=torch.tensor([3, 0]), goal=torch.tensor([-4, 0]))
        env.add_ado(position=torch.tensor([-3, 2]), goal=torch.tensor([1, 5]))

        # Build computation graph to detach later on. Then check whether the graph has been been built by checking
        # for gradient availability.
        ado_action = torch.rand(2)
        ado_action.requires_grad = True
        env.ados[0].update(ado_action, dt=env.dt)

        if env.is_differentiable_wrt_ego:
            assert env.ados[0].position.grad_fn is not None

        # Detach computation graph.
        env.detach()
        assert env.ados[0].position.grad_fn is None

    @staticmethod
    def test_copy(environment_class: mantrap.environment.base.GraphBasedEnvironment.__class__):
        ego_init_pos = torch.tensor([-5, 0])
        ados_init_pos = torch.stack([torch.tensor([1.0, 0.0]), torch.tensor([-6, 2.5])])
        ados_init_vel = torch.stack([torch.tensor([4.2, -1]), torch.tensor([-7, -2.0])])
        ados_goal = torch.stack([torch.zeros(2), torch.ones(2)])

        # Create example environment scene to  copy later on. Then copy the example environment.
        env = environment_class(ego_type=mantrap.agents.IntegratorDTAgent, ego_position=ego_init_pos)
        env.add_ado(position=ados_init_pos[0], velocity=ados_init_vel[0], goal=ados_goal[0])
        env.add_ado(position=ados_init_pos[1], velocity=ados_init_vel[1], goal=ados_goal[1])
        env_copy = env.copy()

        # Test equality of basic environment properties and states.
        assert env.name == env_copy.name
        assert env.time == env_copy.time
        assert env.dt == env_copy.dt

        assert env.same_initial_conditions(other=env_copy)
        assert env.ego == env_copy.ego
        for i in range(env.num_ados):  # agents should be equal and in the same order
            assert env.ados[i] == env_copy.ados[i]
            assert env.ado_ids[i] == env_copy.ado_ids[i]
        ego_state_original, ado_states_original = env.states()
        ego_state_copy,  ado_states_copy = env_copy.states()
        assert torch.all(torch.eq(ego_state_original, ego_state_copy))
        assert torch.all(torch.eq(ado_states_original, ado_states_copy))

        # Test broken link between `env` and `env_copy`, i.e. when I change env_copy, then the original
        # environment remains unchanged.
        env_copy.step(ego_action=torch.ones(2))  # does not matter here anyways
        ego_state_original, ado_states_original = env.states()
        ego_state_copy,  ado_states_copy = env_copy.states()
        assert not torch.all(torch.eq(ego_state_original, ego_state_copy))
        assert not torch.all(torch.eq(ado_states_original, ado_states_copy))

    @staticmethod
    def test_states(environment_class: mantrap.environment.base.GraphBasedEnvironment.__class__):
        ego_position = torch.tensor([-5, 0])
        env = environment_class(ego_type=mantrap.agents.IntegratorDTAgent, ego_position=ego_position)
        env.add_ado(position=torch.tensor([3, 0]), velocity=torch.rand(2), goal=torch.rand(2))
        env.add_ado(position=torch.tensor([-4, 2]), velocity=torch.ones(2), goal=torch.rand(2))

        ego_state, ado_states = env.states()
        assert mantrap.utility.shaping.check_ego_state(ego_state, enforce_temporal=True)
        assert mantrap.utility.shaping.check_ado_states(ado_states, enforce_temporal=True)

        # The first entry of every predicted trajectory should be the current state, check that.
        ado_trajectories = env.predict_wo_ego(t_horizon=2)
        assert torch.allclose(ado_trajectories[:, 0, 0, :], ado_states[:, 0:2], atol=0.01)
        ado_samples = env.sample_wo_ego(t_horizon=2, num_samples=1)
        assert torch.allclose(ado_samples[:, 0, 0, 0, :], ado_states[:, 0:2], atol=0.01)

        # Test that the states are the same as the states of actual agents.
        assert torch.all(torch.eq(ego_state, env.ego.state_with_time))
        for m_ado, ado in enumerate(env.ados):
            assert torch.all(torch.eq(ado_states[m_ado, :], ado.state_with_time))


###########################################################################
# Test - Social Forces Environment ########################################
###########################################################################
@pytest.mark.parametrize("goal_position", [torch.tensor([2.0, 2.0]), torch.tensor([0.0, -2.0])])
def test_social_forces_single_ado_prediction(goal_position: torch.Tensor):
    env = mantrap.environment.SocialForcesEnvironment()
    env.add_ado(goal=goal_position, position=torch.tensor([-1, -5]), velocity=torch.ones(2) * 0.8)

    trajectory_samples = env.sample_wo_ego(t_horizon=100, num_samples=100)
    trajectory = torch.mean(trajectory_samples, dim=1).squeeze()
    assert torch.isclose(trajectory[-1][0], goal_position[0], atol=0.5)
    assert torch.isclose(trajectory[-1][1], goal_position[1], atol=0.5)


def test_social_forces_static_ado_pair_prediction():
    env = mantrap.environment.SocialForcesEnvironment()
    env.add_ado(goal=torch.zeros(2), position=torch.tensor([-1, 0]), velocity=torch.tensor([0.1, 0]))
    env.add_ado(goal=torch.zeros(2), position=torch.tensor([1, 0]), velocity=torch.tensor([-0.1, 0]))

    trajectories = env.sample_wo_ego(t_horizon=10, num_samples=100)
    trajectories = torch.mean(trajectories, dim=1).squeeze()
    # Due to the repulsive of the agents between each other, they cannot both go to their goal position (which is
    # the same for both of them). Therefore the distance must be larger then zero basically, otherwise the repulsive
    # force would not act (or act attractive instead of repulsive).
    assert torch.norm(trajectories[0, -1, 0:1] - trajectories[1, -1, 0:1]) > 1e-3


###########################################################################
# Test - Potential Field Environment ######################################
###########################################################################
@pytest.mark.parametrize(
    "pos_1, pos_2",
    [
        (torch.tensor([1, 1]), torch.tensor([2, 2])),
        (torch.tensor([2, 0]), torch.tensor([4, 0])),
        (torch.tensor([0, 2]), torch.tensor([0, 4])),
    ],
)
def test_potential_field_forces(pos_1: torch.Tensor, pos_2: torch.Tensor):
    env_1 = mantrap.environment.PotentialFieldEnvironment(mantrap.agents.IntegratorDTAgent, ego_position=pos_1)
    env_2 = mantrap.environment.PotentialFieldEnvironment(mantrap.agents.IntegratorDTAgent, ego_position=pos_2)

    t_horizon = 4
    mus = torch.zeros((2, t_horizon + 1, env_1.num_modes, 2))
    sigmas = torch.zeros((2, t_horizon + 1, env_1.num_modes, 2))
    grads = torch.zeros((2, t_horizon, 2))
    for i, env in enumerate([env_1, env_2]):
        env.add_ado(position=torch.zeros(2), velocity=torch.zeros(2))

        ego_controls = torch.zeros((4, 2))
        ego_controls.requires_grad = True
        ego_trajectory = env.ego.unroll_trajectory(ego_controls, dt=env.dt)
        dist_dict = env.compute_distributions(ego_trajectory=ego_trajectory)

        mus[i, :, :, :] = dist_dict[env.ado_ids[0]].mean
        sigmas[i, :, :, :] = dist_dict[env.ado_ids[0]].variance
        grads[i, :, :] = torch.autograd.grad(torch.norm(mus[i, -1, :, :]), ego_controls)[0]

    # The interaction "force" is distance based, so more distant agents should affect a smaller "force".
    # Due to a larger induced force differences between the particle parameters are larger, so that the
    # uncertainty grows larger as well. (0 ==> "close" ego; 1 ==> "far" ego)
    assert torch.all(torch.ge(torch.norm(mus[0, :, :, :], dim=2), torch.norm(mus[1, :, :, :], dim=2)))
    # assert torch.all(torch.norm(sigmas[0, :, :], dim=1) >= torch.norm(sigmas[1, :, :], dim=1))

    # Similarly the gradient should be larger, the closer the ego is since its "impact" increases.
    # assert torch.all(torch.ge(torch.norm(grads[0, :, :], dim=1), torch.norm(grads[1, :, :], dim=1)))

    # When the delta position is uni-directional, so e.g. just in x-position, the force as well as the gradient
    # should point only in this direction.
    for i, pos in enumerate([pos_1, pos_2]):
        for k in [0, 1]:
            if pos[k] == 0:
                assert torch.allclose(grads[i, :, k], torch.zeros(t_horizon))


###########################################################################
# Test - Kalman Environment ###############################################
###########################################################################
def test_kalman_distributions():
    env = mantrap.environment.KalmanEnvironment()
    x0, y0 = 3.7, -5.1
    vx, vy = -1.0, 0.9
    env.add_ado(position=torch.tensor([x0, y0]), velocity=torch.tensor([vx, vy]))

    t_horizon = 4
    dist_dict = env.compute_distributions_wo_ego(t_horizon=t_horizon)
    mean = dist_dict[env.ado_ids[0]].mean
    variance = dist_dict[env.ado_ids[0]].variance

    assert torch.allclose(mean[:, 0, 0], torch.linspace(x0, x0 + vx * t_horizon * env.dt, steps=t_horizon + 1))
    assert torch.allclose(mean[:, 0, 1], torch.linspace(y0, y0 + vy * t_horizon * env.dt, steps=t_horizon + 1))

    variance_diff = (variance[1:, :, :] - variance[:-1, :, :]).squeeze()
    assert torch.all(variance_diff >= 0)  # variance is strictly increasing over time


###########################################################################
# Test - Trajectron Environment ###########################################
###########################################################################
def test_trajectron_wo_prediction():
    env = mantrap.environment.Trajectron(mantrap.agents.DoubleIntegratorDTAgent,
                                         ego_position=torch.zeros(2))
    env.add_ado(position=torch.tensor([4, 4]), velocity=torch.tensor([0, -1]))

    samples_wo = env.sample_wo_ego(t_horizon=10, num_samples=5)
    assert mantrap.utility.shaping.check_ado_samples(samples_wo, ados=env.num_ados, num_samples=5)

    ego_controls = torch.zeros((10, 2))
    samples_with = env.sample_w_controls(ego_controls, num_samples=5)
    assert mantrap.utility.shaping.check_ado_samples(samples_with, ados=env.num_ados, num_samples=5)


###########################################################################
# Test - SGAN Environment #################################################
###########################################################################
# def absolute_to_relative(trajectory: torch.Tensor) -> torch.Tensor:
#     t_horizon, num_agents, dim = trajectory.shape
#     trajectory_rel = trajectory - trajectory[0, :, :].unsqueeze(dim=0)
#     trajectory_rel = trajectory_rel[1:, :, :] - trajectory_rel[:-1, :, :]
#     return torch.cat((torch.zeros(1, num_agents, dim), trajectory_rel), dim=0)


###########################################################################
# Test - ORCA Environment #################################################
###########################################################################
# Comparison to original implementation of ORCA which can be found in
# external code directory and uses the following parameters:
# orca_rad = 1.0
# orca_dt = 10.0
# sim_dt = 0.25
# sim_speed_max = 4.0
# @pytest.mark.xfail(raises=AssertionError)
# def test_orca_single_agent():
#     env = mantrap.environment.ORCAEnvironment(dt=0.25)
#     env.add_ado(position=torch.zeros(2), velocity=torch.zeros(2), goal=torch.ones(2) * 4)
#
#     pos_expected = torch.tensor([[0, 0], [0.70710, 0.70710], [1.4142, 1.4142], [2.1213, 2.1213], [2.8284, 2.8284]])
#     ado_trajectories = env.sample_wo_ego(t_horizon=pos_expected.shape[0])
#     assert torch.isclose(torch.norm(ado_trajectories[0, 0, :, 0:2] - pos_expected), torch.zeros(1), atol=0.1)
#
#
# @pytest.mark.xfail(raises=AssertionError)
# def test_orca_two_agents():
#     env = mantrap.environment.ORCAEnvironment(dt=0.25)
#     env.add_ado(position=torch.tensor([-5, 0.1]), velocity=torch.zeros(2), goal=torch.tensor([5, 0]))
#     env.add_ado(position=torch.tensor([5, -0.1]), velocity=torch.zeros(2), goal=torch.tensor([-5, 0]))
#
#     pos_expected = torch.tensor(
#         [
#             [
#                 [-5, 0.1],
#                 [-4.8998, 0.107995],
#                 [-4.63883, 0.451667],
#                 [-3.65957, 0.568928],
#                 [-2.68357, 0.6858],
#                 [-1.7121, 0.802128],
#                 [-0.747214, 0.917669],
#                 [0.207704, 1.03202],
#                 [1.18529, 0.821493],
#                 [2.16288, 0.61097],
#             ],
#             [
#                 [5, -0.1],
#                 [4.8998, -0.107995],
#                 [4.63883, -0.451667],
#                 [3.65957, -0.568928],
#                 [2.68357, -0.6858],
#                 [1.7121, -0.802128],
#                 [0.747214, -0.917669],
#                 [-0.207704, -1.03202],
#                 [-1.18529, -0.821493],
#                 [-2.16288, -0.61097],
#             ],
#         ]
#     ).view(2, 1, -1, 2)
#     ado_trajectories = env.sample_wo_ego(t_horizon=pos_expected.shape[2])
#     assert torch.isclose(torch.norm(ado_trajectories[:, :, :, 0:2] - pos_expected), torch.zeros(1), atol=0.1)
