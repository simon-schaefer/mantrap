import pytest
import torch

import mantrap.agents
import mantrap.utility.maths


###########################################################################
# Tests - All Agents ######################################################
###########################################################################
@pytest.mark.parametrize("agent_class", [mantrap.agents.IntegratorDTAgent,
                                         mantrap.agents.DoubleIntegratorDTAgent])
class TestAgent:

    @staticmethod
    def test_dynamics(agent_class: mantrap.agents.base.DTAgent.__class__):
        control = torch.rand(2)
        agent = agent_class(position=torch.rand(2), velocity=torch.rand(2))
        state_next = agent.dynamics(state=agent.state_with_time, action=control, dt=1.0)
        control_output = agent.inverse_dynamics(state_previous=agent.state, state=state_next, dt=1.0)
        assert torch.all(torch.isclose(control, control_output))

    @staticmethod
    def test_history(agent_class: mantrap.agents.base.DTAgent.__class__):
        agent = agent_class(position=torch.tensor([-1, 4]), velocity=torch.ones(2))
        for _ in range(4):
            agent.update(action=torch.ones(2), dt=1.0)
        assert len(agent.history.shape) == 2
        assert agent.history.shape[0] == 5
        assert agent.history.shape[1] == 5

    @staticmethod
    def test_reset(agent_class: mantrap.agents.base.DTAgent.__class__):
        agent = agent_class(position=torch.tensor([5, 6]))
        agent.reset(state=torch.tensor([1, 5, 4, 2, 1.0]), history=None)
        assert torch.all(torch.eq(agent.position, torch.tensor([1, 5]).float()))
        assert torch.all(torch.eq(agent.velocity, torch.tensor([4, 2]).float()))

    @staticmethod
    def test_rolling(agent_class: mantrap.agents.base.DTAgent.__class__):
        agent = agent_class(position=torch.zeros(2))
        controls = torch.tensor([[1, 1], [2, 2], [4, 4]]).float()
        trajectory = agent.unroll_trajectory(controls, dt=1.0)
        assert torch.all(torch.isclose(controls, agent.roll_trajectory(trajectory, dt=1.0)))

    @staticmethod
    def test_initialization(agent_class: mantrap.agents.base.DTAgent.__class__):
        # history initial value = None.
        agent = agent_class(position=torch.zeros(2), velocity=torch.zeros(2), history=None)
        assert torch.all(torch.eq(agent.history, torch.zeros((1, 5))))

        # history initial value != None
        history = torch.tensor([5, 6, 0, 0, -1]).view(1, 5).float()
        history_exp = torch.tensor([[5, 6, 0, 0, -1],
                                    [1, 1, 0, 0, 0]]).float()
        agent = agent_class(position=torch.ones(2), velocity=torch.zeros(2), history=history)
        assert torch.all(torch.eq(agent.history, history_exp))

    @staticmethod
    def test_update(agent_class: mantrap.agents.base.DTAgent.__class__):
        """Test agent `update()` using the `dynamics()` which has been tested independently so is fairly safe
        to use for testing since it grants generality over agent types. """
        state_init = torch.rand(4)
        control_input = torch.rand(2)
        agent = agent_class(position=state_init[0:2], velocity=state_init[2:4])
        state_next = agent.dynamics(state=agent.state_with_time, action=control_input, dt=1.0)
        agent.update(control_input, dt=1.0)

        assert torch.all(torch.isclose(agent.position, state_next[0:2]))
        assert torch.all(torch.isclose(agent.velocity, state_next[2:4]))
        assert torch.all(torch.isclose(agent.history[0, 0:4], state_init))
        assert torch.all(torch.isclose(agent.history[1, :], state_next))

    @staticmethod
    def test_forward_reachability(agent_class: mantrap.agents.base.DTAgent.__class__):
        agent = agent_class(position=torch.rand(2) * 5, velocity=torch.rand(2) * 2)
        num_samples, time_steps = 100, 4
        control_min, control_max = agent.control_limits()
        control_samples = torch.linspace(start=control_min, end=control_max, steps=num_samples)

        def _compute_end_points(control_x: float, control_y: float) -> torch.Tensor:
            controls = torch.ones((time_steps, 2)).__mul__(torch.tensor([control_x, control_y]))
            controls = agent.make_controls_feasible(controls)
            return agent.unroll_trajectory(controls=controls, dt=1.0)[-1, 0:2]

        # Compute the forward reachability bounds numerically by integrating trajectories at the bound
        # of feasible control inputs, i.e. since 2D: (x, y_max), (x, y_min), (x_max, y), (x_min, y).
        boundary_numeric = torch.zeros((4*num_samples, 2))
        for i, control_sample in enumerate(control_samples):
            boundary_numeric[i + 0*num_samples, :] = _compute_end_points(control_sample, control_max)
            boundary_numeric[i + 1*num_samples, :] = _compute_end_points(control_sample, control_min)
            boundary_numeric[i + 2*num_samples, :] = _compute_end_points(control_max, control_sample)
            boundary_numeric[i + 3*num_samples, :] = _compute_end_points(control_min, control_sample)

        # Use the agents method to compute the same boundary (analytically).
        boundary = agent.reachability_boundary(time_steps, dt=1.0)

        # Derive numeric bounds properties. This is not sufficient proof for being a circle but at
        # least a necessary condition.
        if type(boundary) == mantrap.utility.maths.Circle:
            min_x, min_y = min(boundary_numeric[:, 0]), min(boundary_numeric[:, 1])
            max_x, max_y = max(boundary_numeric[:, 0]), max(boundary_numeric[:, 1])
            radius_numeric = (max_x - min_x) / 2
            center_numeric = torch.tensor([min_x + radius_numeric, min_y + radius_numeric])

            assert torch.all(torch.isclose(boundary.center, center_numeric))  # center ?
            assert torch.isclose((max_y - min_y) / 2, radius_numeric, atol=0.1)  # is circle ?
            assert torch.isclose(torch.tensor(boundary.radius), radius_numeric, atol=0.1)  # same circle ?

    @staticmethod
    def test_feasibility_check(agent_class: mantrap.agents.base.DTAgent.__class__):
        # Only check the feasibility controls function since the feasibility trajectory function simply
        # adds a rolling of the trajectory to a set of controls, which has been tested before.
        agent = agent_class(position=torch.rand(2) * 5, velocity=torch.rand(2) * 2)

        controls = torch.zeros((20, 2))
        assert agent.check_feasibility_controls(controls=controls)

        controls[5, 0] = 100.0  # should be far off every maximum
        assert not agent.check_feasibility_controls(controls=controls)


###########################################################################
# Test - Single Integrator ################################################
###########################################################################
@pytest.mark.parametrize(
    "position, velocity, control, position_expected, velocity_expected",
    [
        (torch.tensor([1, 0]), torch.zeros(2), torch.zeros(2), torch.tensor([1, 0]), torch.zeros(2)),
        (torch.tensor([1, 0]), torch.tensor([2, 3]), torch.zeros(2), torch.tensor([1, 0]), torch.tensor([0, 0])),
        (torch.tensor([1, 0]), torch.zeros(2), torch.tensor([2, 3]), torch.tensor([3, 3]), torch.tensor([2, 3])),
    ],
)
def test_dynamics_single_integrator(
    position: torch.Tensor,
    velocity: torch.Tensor,
    control: torch.Tensor,
    position_expected: torch.Tensor,
    velocity_expected: torch.Tensor,
):
    agent = mantrap.agents.IntegratorDTAgent(position=position, velocity=velocity)
    state_next = agent.dynamics(state=agent.state_with_time, action=control, dt=1.0)
    assert torch.all(torch.isclose(state_next[0:2], position_expected.float()))
    assert torch.all(torch.isclose(state_next[2:4], velocity_expected.float()))


@pytest.mark.parametrize(
    "position, velocity, state_previous, control_expected",
    [
        (torch.tensor([1, 0]), torch.zeros(2), torch.zeros(5), torch.tensor([1, 0])),
        (torch.tensor([1, 0]), torch.tensor([2, 3]), torch.zeros(5), torch.tensor([1, 0])),
        (torch.tensor([1, 0]), torch.zeros(2), torch.tensor([2, 0, 0, 0, -1.0]), torch.tensor([-1, 0])),
    ],
)
def test_inv_dynamics_single_integrator(
    position: torch.Tensor,
    velocity: torch.Tensor,
    state_previous: torch.Tensor,
    control_expected: torch.Tensor,
):
    agent = mantrap.agents.IntegratorDTAgent(position=position, velocity=velocity)
    control = agent.inverse_dynamics(state=agent.state, state_previous=state_previous, dt=1.0)
    assert torch.all(torch.isclose(control, control_expected.float()))


@pytest.mark.parametrize(
    "position, velocity, dt, n",
    [
        (torch.tensor([-5.0, 0.0]), torch.tensor([1.0, 0.0]), 1, 10),

    ]
)
def test_unrolling(position: torch.Tensor, velocity: torch.Tensor, dt: float, n: int):
    ego = mantrap.agents.IntegratorDTAgent(position=position, velocity=velocity)
    policy = torch.cat((torch.ones(n, 1) * velocity[0], torch.ones(n, 1) * velocity[1]), dim=1)
    ego_trajectory = ego.unroll_trajectory(controls=policy, dt=dt)

    ego_trajectory_x_exp = torch.linspace(position[0].item(), position[0].item() + velocity[0].item() * n * dt, n + 1)
    ego_trajectory_y_exp = torch.linspace(position[1].item(), position[1].item() + velocity[1].item() * n * dt, n + 1)

    assert torch.all(torch.eq(ego_trajectory[:, 0], ego_trajectory_x_exp))
    assert torch.all(torch.eq(ego_trajectory[:, 1], ego_trajectory_y_exp))
    assert torch.all(torch.eq(ego_trajectory[:, 2], torch.ones(n + 1) * velocity[0]))
    assert torch.all(torch.eq(ego_trajectory[:, 3], torch.ones(n + 1) * velocity[1]))

    print(ego_trajectory[:, 4], torch.linspace(0, n, n + 1))

    assert torch.all(torch.eq(ego_trajectory[:, 4], torch.linspace(0, n, n + 1)))


###########################################################################
# Test - Double Integrator ################################################
###########################################################################
@pytest.mark.parametrize(
    "position, velocity, control, position_expected, velocity_expected",
    [
        (torch.tensor([1, 0]), torch.zeros(2), torch.zeros(2), torch.tensor([1, 0]), torch.zeros(2)),
        (torch.tensor([1, 0]), torch.tensor([2, 3]), torch.zeros(2), torch.tensor([3, 3]), torch.tensor([2, 3])),
        (torch.tensor([1, 0]), torch.zeros(2), torch.tensor([2, 3]), torch.tensor([1, 0.0]), torch.tensor([2, 3])),
    ],
)
def test_dynamics_double_integrator(
    position: torch.Tensor,
    velocity: torch.Tensor,
    control: torch.Tensor,
    position_expected: torch.Tensor,
    velocity_expected: torch.Tensor,
):
    agent = mantrap.agents.DoubleIntegratorDTAgent(position=position, velocity=velocity)
    state_next = agent.dynamics(state=agent.state_with_time, action=control, dt=1.0)

    assert torch.all(torch.isclose(state_next[0:2], position_expected.float()))
    assert torch.all(torch.isclose(state_next[2:4], velocity_expected.float()))


@pytest.mark.parametrize(
    "position, velocity, state_previous, control_expected",
    [
        (torch.tensor([1, 0]), torch.zeros(2), torch.zeros(5), torch.tensor([1, 0])),
        (torch.tensor([1, 0]), torch.tensor([2, 3]), torch.zeros(5), torch.tensor([1, 0])),
        (torch.tensor([1, 0]), torch.zeros(2), torch.tensor([2, 0, 0, 0, -1.0]), torch.tensor([-1, 0])),
    ],
)
def test_inv_dynamics_double_integrator(
    position: torch.Tensor,
    velocity: torch.Tensor,
    state_previous: torch.Tensor,
    control_expected: torch.Tensor,
):
    agent = mantrap.agents.IntegratorDTAgent(position=position, velocity=velocity)
    control = agent.inverse_dynamics(state=agent.state, state_previous=state_previous, dt=1.0)
    assert torch.all(torch.isclose(control, control_expected.float()))
