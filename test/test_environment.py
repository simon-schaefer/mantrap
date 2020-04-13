from collections import namedtuple
from typing import List

import numpy as np
import pytest
import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.constants import *
from mantrap.environment.environment import GraphBasedEnvironment
from mantrap.environment import PotentialFieldEnvironment, SocialForcesEnvironment, Trajectron
from mantrap.utility.maths import Distribution, DirecDelta
from mantrap.utility.primitives import straight_line
from mantrap.utility.shaping import check_ado_trajectories, check_ado_states, check_ego_state


###########################################################################
# Tests - All Environment #################################################
###########################################################################
# In order to test the functionality of the environment in a standardized way
@pytest.mark.parametrize(
    "environment_class", (SocialForcesEnvironment, PotentialFieldEnvironment, Trajectron)
)
class TestEnvironment:

    @staticmethod
    def test_initialization(environment_class: GraphBasedEnvironment.__class__):
        env = environment_class(ego_type=IntegratorDTAgent, ego_kwargs={"position": torch.tensor([4, 6])})
        assert torch.all(torch.eq(env.ego.position, torch.tensor([4, 6]).float()))
        assert env.num_ados == 0
        assert env.time == 0.0
        env.add_ado(position=torch.tensor([6, 7]), velocity=torch.zeros(2), num_modes=1)

        assert torch.all(torch.eq(env.ghosts[0].agent.position, torch.tensor([6, 7]).float()))
        assert torch.all(torch.eq(env.ghosts[0].agent.velocity, torch.zeros(2)))

    @staticmethod
    def test_step(environment_class: GraphBasedEnvironment.__class__):
        ado_init_position = torch.zeros(2)
        ado_init_velocity = torch.zeros(2)
        ego_init_position = torch.tensor([-4, 6])
        env = environment_class(ego_type=IntegratorDTAgent, ego_kwargs={"position": ego_init_position})

        env.add_ado(position=ado_init_position, velocity=ado_init_velocity)
        assert env.num_ados == 1
        assert env.num_modes == 1
        assert env.num_ghosts == 1

        t_horizon = 5
        ego_controls = torch.stack([torch.tensor([1, 0])] * t_horizon)
        ego_trajectory = env.ego.unroll_trajectory(controls=ego_controls, dt=env.dt)
        for t in range(t_horizon):
            ado_t, ego_t = env.step(ego_control=ego_controls[t:t+1])

            # Check dimensions of outputted ado and ego states.
            assert ado_t.numel() == 5
            assert ado_t.shape == (1, 5)
            assert ego_t.numel() == 5

            # While the exact value of the ado agent's states depends on the environment dynamics used, all of them
            # are based on the ego state (control), which is thought to be enforced while forwarding the environment.
            assert all(torch.eq(ego_t, ego_trajectory[t+1, :]))

    @staticmethod
    def test_step_reset(environment_class: GraphBasedEnvironment.__class__):
        env = environment_class(ego_type=IntegratorDTAgent, ego_kwargs={"position": torch.tensor([-4, 6])})
        env.add_ado(position=torch.zeros(2), velocity=torch.zeros(2), num_modes=1)
        env.add_ado(position=torch.ones(2), velocity=torch.zeros(2), num_modes=1)

        ego_next_state = torch.rand(5)
        ado_next_states = torch.rand(env.num_ados, 5)
        env.step_reset(ego_state_next=ego_next_state, ado_states_next=ado_next_states)

        assert torch.all(torch.eq(env.ego.state_with_time, ego_next_state))
        for i in range(env.num_ghosts):
            assert torch.all(torch.eq(env.ghosts[i].agent.state_with_time, ado_next_states[i]))

    @staticmethod
    def test_prediction_trajectories_shape(environment_class: GraphBasedEnvironment.__class__):
        num_modes = 2
        t_horizon = 4

        env = environment_class()
        history = torch.stack(5 * [torch.tensor([1, 0, 0, 0, 0])])
        env.add_ado(goal=torch.ones(2), position=torch.tensor([-1, 0]), num_modes=num_modes, history=history)
        env.add_ado(goal=torch.zeros(2), position=torch.tensor([1, 0]), num_modes=num_modes, history=history)

        ado_trajectories = env.predict_wo_ego(t_horizon=t_horizon)
        assert check_ado_trajectories(ado_trajectories, t_horizon=t_horizon, modes=num_modes, ados=2)

    @staticmethod
    def test_build_connected_graph(environment_class: GraphBasedEnvironment.__class__):
        env = environment_class(ego_type=IntegratorDTAgent, ego_kwargs={"position": torch.tensor([-5, 0])})
        env.add_ado(position=torch.tensor([3, 0]), velocity=torch.zeros(2), goal=torch.tensor([-4, 0]), num_modes=2)
        env.add_ado(position=torch.tensor([5, 0]), velocity=torch.zeros(2), goal=torch.tensor([-4, 0]), num_modes=2)
        env.add_ado(position=torch.tensor([10, 0]), velocity=torch.zeros(2), goal=torch.tensor([-4, 0]), num_modes=2)

        prediction_horizon = 10
        trajectory = torch.zeros((prediction_horizon, 4))  # does not matter here anyway
        graphs = env.build_connected_graph(ego_trajectory=trajectory)

        assert env.check_graph(graph=graphs, include_ego=True, t_horizon=prediction_horizon)

    @staticmethod
    def test_ego_graph_updates(environment_class: GraphBasedEnvironment.__class__):
        position = torch.tensor([-5, 0])
        goal = torch.tensor([5, 0])
        path = straight_line(start_pos=position, end_pos=goal, steps=11)

        ego_kwargs = {"position": position, "velocity": torch.zeros(2)}
        env = environment_class(ego_type=IntegratorDTAgent, ego_kwargs=ego_kwargs)
        env.add_ado(position=torch.tensor([3, 0]), velocity=torch.zeros(2), goal=torch.zeros(2))

        graphs = env.build_connected_graph(ego_trajectory=torch.cat((path, torch.zeros((11, 2))), dim=1))
        for k in range(path.shape[0]):
            assert torch.all(torch.eq(path[k, :], graphs[f"{ID_EGO}_{k}_{GK_POSITION}"]))

    @staticmethod
    def test_ghost_sorting(environment_class: GraphBasedEnvironment.__class__):
        env = environment_class()
        weights_initial = [0.08, 0.1, 0.8, 0.02]
        env.add_ado(position=torch.zeros(2), num_modes=4, weights=torch.tensor(weights_initial))

        ghost_weights = [ghost.weight for ghost in env.ghosts]
        assert ghost_weights == list(reversed(sorted(weights_initial)))  # sorted increasing values per default

    @staticmethod
    def test_detaching(environment_class: GraphBasedEnvironment.__class__):
        env = environment_class(ego_type=IntegratorDTAgent, ego_kwargs={"position": torch.tensor([-5, 0])})
        env.add_ado(position=torch.tensor([3, 0]), velocity=torch.zeros(2), goal=torch.tensor([-4, 0]), num_modes=2)

        # Build computation graph to detach later on. Then check whether the graph has been been built by checking
        # for gradient availability.
        ego_control = torch.ones((1, 2))
        ego_control.requires_grad = True
        _, ado_controls, weights = env.predict_w_controls(ego_controls=ego_control, return_more=True)
        for j in range(env.num_ghosts):
            ado_id, _ = env.split_ghost_id(ghost_id=env.ghosts[j].id)
            i_ado = env.index_ado_id(ado_id=ado_id)
            env._ado_ghosts[j].agent.update(action=ado_controls[i_ado, 0, 0, :], dt=env.dt)
        assert env.ghosts[0].agent.position.grad_fn is not None

        # Detach computation graph.
        env.detach()
        assert env.ghosts[0].agent.position.grad_fn is None

    @staticmethod
    def test_copy(environment_class: GraphBasedEnvironment.__class__):
        ego_init_pos = torch.tensor([-5, 0])
        ados_init_pos = torch.stack([torch.tensor([1.0, 0.0]), torch.tensor([-6, 2.5])])
        ados_init_vel = torch.stack([torch.tensor([4.2, -1]), torch.tensor([-7, -2.0])])
        ados_goal = torch.stack([torch.zeros(2), torch.ones(2)])
        num_modes = 2

        # Create example environment scene to  copy later on. Then copy the example environment.
        env = environment_class(ego_type=IntegratorDTAgent, ego_kwargs={"position": ego_init_pos})
        env.add_ado(position=ados_init_pos[0], velocity=ados_init_vel[0], goal=ados_goal[0], num_modes=num_modes)
        env.add_ado(position=ados_init_pos[1], velocity=ados_init_vel[1], goal=ados_goal[1], num_modes=num_modes)
        env_copy = env.copy()

        # Test equality of basic environment properties and states.
        assert env.environment_name == env_copy.environment_name
        assert env.time == env_copy.time
        assert env.dt == env_copy.dt

        assert env.same_initial_conditions(other=env_copy)
        assert env.num_ghosts == env_copy.num_ghosts
        assert env.num_modes == env_copy.num_modes
        assert env.ego == env_copy.ego
        for i in range(env.num_ghosts):  # agents should be equal and in the same order
            assert env.ghosts[i].agent == env_copy.ghosts[i].agent
            assert env.ghosts[i].weight == env_copy.ghosts[i].weight
            assert env.ghosts[i].id == env_copy.ghosts[i].id
            assert env.ghosts[i].params == env_copy.ghosts[i].params
        ego_state_original, ado_states_original = env.states()
        ego_state_copy,  ado_states_copy = env_copy.states()
        assert torch.all(torch.eq(ego_state_original, ego_state_copy))
        assert torch.all(torch.eq(ado_states_original, ado_states_copy))

        # Test whether predictions are equal in both environments.
        ego_controls = torch.rand((5, 2))
        traj_original = env.predict_w_controls(ego_controls=ego_controls)
        traj_copy = env.predict_w_controls(ego_controls=ego_controls)
        assert torch.all(torch.eq(traj_original, traj_copy))

        # Test broken link between `env` and `env_copy`, i.e. when I change env_copy, then the original environment
        # remains unchanged.
        env_copy.step(ego_control=torch.ones(1, 2))  # does not matter here anyways
        ego_state_original, ado_states_original = env.states()
        ego_state_copy,  ado_states_copy = env_copy.states()
        assert not torch.all(torch.eq(ego_state_original, ego_state_copy))
        assert not torch.all(torch.eq(ado_states_original, ado_states_copy))

    @staticmethod
    def test_states(environment_class: GraphBasedEnvironment.__class__):
        env = environment_class(ego_type=IntegratorDTAgent, ego_kwargs={"position": torch.tensor([-5, 0])})
        env.add_ado(position=torch.tensor([3, 0]), velocity=-torch.ones(2), goal=torch.tensor([-4, 0]), num_modes=2)
        env.add_ado(position=torch.tensor([-4, 2]), velocity=torch.ones(2), goal=torch.tensor([-4, 0]), num_modes=2)

        ego_state, ado_states = env.states()
        assert check_ego_state(x=ego_state, enforce_temporal=True)
        assert check_ado_states(x=ado_states, enforce_temporal=True)

        # The first entry of every predicted trajectory should be the current state, check that.
        ado_trajectories = env.predict_wo_ego(t_horizon=2)
        for m_mode in range(env.num_modes):
            assert torch.all(torch.eq(ado_states, ado_trajectories[:, m_mode, 0, :]))

        # Test that the states are the same as the states of actual agents.
        assert torch.all(torch.eq(ego_state, env.ego.state_with_time))
        for ghost in env.ghosts:
            m_ado, _ = env.convert_ghost_id(ghost_id=ghost.id)
            assert torch.all(torch.eq(ado_states[m_ado, :], ghost.agent.state_with_time))


###########################################################################
# Test - Social Forces Environment ########################################
###########################################################################
@pytest.mark.parametrize("goal_position", [torch.tensor([2.0, 2.0]), torch.tensor([0.0, -2.0])])
def test_social_forces_single_ado_prediction(goal_position: torch.Tensor):
    env = SocialForcesEnvironment()
    env.add_ado(goal=goal_position, position=torch.tensor([-1, -5]), velocity=torch.ones(2) * 0.8, num_modes=1)

    trajectory = torch.squeeze(env.predict_wo_ego(t_horizon=100))
    assert torch.isclose(trajectory[-1][0], goal_position[0], atol=0.5)
    assert torch.isclose(trajectory[-1][1], goal_position[1], atol=0.5)


def test_social_forces_static_ado_pair_prediction():
    env = SocialForcesEnvironment()
    env.add_ado(goal=torch.zeros(2), position=torch.tensor([-1, 0]), velocity=torch.tensor([0.1, 0]), num_modes=1)
    env.add_ado(goal=torch.zeros(2), position=torch.tensor([1, 0]), velocity=torch.tensor([-0.1, 0]), num_modes=1)

    trajectories = env.predict_wo_ego(t_horizon=100)
    # Due to the repulsive of the agents between each other, they cannot both go to their goal position (which is
    # the same for both of them). Therefore the distance must be larger then zero basically, otherwise the repulsive
    # force would not act (or act attractive instead of repulsive).
    assert torch.norm(trajectories[0, -1, 0:1] - trajectories[1, -1, 0:1]) > 1e-3


@pytest.mark.parametrize(
    "pos, vel, num_modes, v0s",
    [(torch.tensor([-1, 0]), torch.tensor([0.1, 0.2]), 2, [DirecDelta(2.3), DirecDelta(1.5)])],
)
def test_social_forces_ghosts_init(pos: torch.Tensor, vel: torch.Tensor, num_modes: int, v0s: List[Distribution]):
    env = SocialForcesEnvironment()
    env.add_ado(goal=torch.zeros(2), position=pos, velocity=vel, num_modes=num_modes, v0s=v0s)

    assert env.num_modes == num_modes
    assert all([env.split_ghost_id(ghost.id)[0] == env.ado_ids[0] for ghost in env.ghosts])
    assert len(env.ghosts) == num_modes

    assert all([type(v0) == DirecDelta for v0 in v0s])  # otherwise hard to compare due to sampling
    v0s_env = [ghost.params["v0"] for ghost in env.ghosts]
    v0s_exp = [v0.mean for v0 in v0s]
    assert set(v0s_env) == set(v0s_exp)


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
    env_1 = PotentialFieldEnvironment(IntegratorDTAgent, {"position": pos_1})
    env_2 = PotentialFieldEnvironment(IntegratorDTAgent, {"position": pos_2})

    forces = torch.zeros((2, 2))
    gradients = torch.zeros((2, 2))
    for i, env in enumerate([env_1, env_2]):
        env.add_ado(position=torch.zeros(2))
        graph = env.build_graph(ego_state=env.ego.state)
        forces[i, :] = graph[f"{env.ghosts[0].id}_0_{GK_CONTROL}"]
        ado_force_norm_0 = graph[f"{env.ghosts[0].id}_0_{GK_OUTPUT}"]
        ego_position_0 = graph[f"{ID_EGO}_0_{GK_POSITION}"]
        gradients[i, :] = torch.autograd.grad(ado_force_norm_0, ego_position_0, retain_graph=True)[0].detach()

    # The force is distance based, so more distant agents should affect a smaller force.
    assert torch.norm(forces[0, :]) > torch.norm(forces[1, :])
    assert torch.norm(gradients[0, :]) > torch.norm(gradients[1, :])
    # When the delta position is uni-directional, so e.g. just in x-position, the force as well as the gradient
    # should point only in this direction.
    for i, pos in enumerate([pos_1, pos_2]):
        for k in [0, 1]:
            if pos[k] == 0:
                assert forces[i, k] == gradients[i, k] == 0.0


###########################################################################
# Test - Trajectron Environment ###########################################
###########################################################################
def test_trajectron_mode_selection():
    num_modes = 2

    # Create distribution object similar to the Trajectron output.
    GMM2D = namedtuple("GMM2D", "mus log_sigmas")
    mus = torch.rand((1, 1, 1, num_modes * 2, 2))
    log_sigmas = torch.stack((torch.linspace(10, 2, steps=num_modes * 2), torch.ones(num_modes * 2))).t()
    log_sigmas = log_sigmas.view((1, 1, 1, num_modes * 2, 2))
    gmm = GMM2D(mus=mus, log_sigmas=log_sigmas)

    # Determine trajectory from distribution. The time horizon `t_horizon` is the prediction horizon of the
    # simulation, however when predicting based on the applied controls of the ego then `t_horizon` is one step longer
    # than the number of control inputs, since the current state is prepended to the trajectories. Similarly, the
    # network prediction is one step longer, therefore we have to pass `t_horizon = 1 + 1 = 2` here.
    _, _, weight_indices = Trajectron.trajectory_from_distribution(
        gmm=gmm, num_output_modes=num_modes, dt=1.0, t_horizon=2, return_more=True, ado_state=torch.zeros(5)
    )

    # Test the obtained weight indices by checking for the smallest element in the GMM's variances, which should be
    # related to the first (highest weight) index.
    log_sigmas = gmm.log_sigmas.permute(0, 1, 3, 2, 4)[0, 0, :, :, 0:2]
    log_sigmas_norm = torch.norm(log_sigmas, dim=2)
    assert np.argmin(log_sigmas_norm) == weight_indices[0]
