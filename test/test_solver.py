import time
from typing import Dict, List, Tuple, Union

import numpy as np
import pytest
import torch

from mantrap.constants import AGENT_SPEED_MAX, CONSTRAINT_MIN_L2_DISTANCE
from mantrap.agents import IntegratorDTAgent
from mantrap.environment import PotentialFieldEnvironment, SocialForcesEnvironment
from mantrap.environment.environment import GraphBasedEnvironment
from mantrap.solver import MonteCarloTreeSearch, SGradSolver, IGradSolver, ORCASolver
from mantrap.solver.solver import Solver
from mantrap.solver.ipopt_solver import IPOPTSolver
from mantrap.utility.shaping import check_ego_trajectory, check_ado_trajectories


def scenario(solver_class: Solver.__class__, **solver_kwargs):
    ego_pos = torch.tensor([-5, 2])
    ego_goal = torch.tensor([1, 1])
    ado_poses = [torch.tensor([3, 2])]

    sim = PotentialFieldEnvironment(IntegratorDTAgent, {"position": ego_pos})
    for pos in ado_poses:
        sim.add_ado(position=pos)
    solver = solver_class(sim, goal=ego_goal, **solver_kwargs)
    z0 = solver.z0s_default(just_one=True).detach().numpy()
    return sim, solver, z0


###########################################################################
# Test - Base Solver ######################################################
###########################################################################
# In order to test the general functionality of the solver base class, the solver is redefined to always follow
# a specific action, e.g. always going in positive x direction with constant and pre-determined speed. Then,
# the results are directly predictable and hence testable.
class ConstantSolver(Solver):
    def determine_ego_controls(self, **solver_kwargs) -> torch.Tensor:
        return self.z0s_default()

    def _optimize(self, z0: torch.Tensor, tag: str, ghost_ids: List[str], **kwargs):
        raise NotImplementedError

    def initialize(self, **solver_params):
        pass

    def num_optimization_variables(self) -> int:
        return self.T

    def z0s_default(self, just_one: bool = False) -> torch.Tensor:
        controls = torch.zeros((self.T, 2))
        controls[0, 0] = 1.0
        return controls

    def z_to_ego_trajectory(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def z_to_ego_controls(self, z: np.ndarray, return_leaf: bool = False) -> torch.Tensor:
        raise NotImplementedError

    ###########################################################################
    # Logging parameters ######################################################
    ###########################################################################
    @property
    def objective_keys(self) -> List[str]:
        return []

    @property
    def constraint_keys(self) -> List[str]:
        return []

    @property
    def cores(self) -> List[str]:
        return ["opt"]


def test_solve():
    sim = PotentialFieldEnvironment(IntegratorDTAgent, {"position": torch.tensor([-8, 0])})
    sim.add_ado(position=torch.tensor([0, 0]), velocity=torch.tensor([-1, 0]))
    solver = ConstantSolver(sim, goal=torch.zeros(2), t_planning=10, objectives=[], constraints=[], verbose=0)

    assert solver.T == 10
    assert torch.all(torch.eq(solver.goal, torch.zeros(2)))

    ego_trajectory_opt, ado_traj = solver.solve(100)
    ado_planned = solver.log["opt/ado_planned"]
    ego_opt_planned = solver.log["opt/ego_planned"]

    # Test output shapes.
    t_horizon_exp = int(8 / sim.dt)  # distance ego position and goal = 8 / velocity 1.0 = 8.0 s
    assert check_ego_trajectory(ego_trajectory_opt, t_horizon=t_horizon_exp)
    assert check_ado_trajectories(ado_traj, t_horizon=t_horizon_exp, ados=sim.num_ados, modes=1)
    assert tuple(ado_planned.shape) == (t_horizon_exp, 1, 1, solver.T + 1, 5)
    assert tuple(ego_opt_planned.shape) == (t_horizon_exp, solver.T + 1, 5)

    # Test ego output trajectory, which can be determined independent from environment since it basically is the
    # unrolled trajectory resulting from applying the same, known control action (determined in `ConstantSolver`)
    # again and again.
    x_path_exp = torch.tensor([-8.0 + 1.0 * sim.dt * k for k in range(t_horizon_exp)])
    assert torch.all(torch.isclose(ego_trajectory_opt[:, 0], x_path_exp, atol=1e-3))
    assert torch.all(torch.eq(ego_trajectory_opt[:, 1], torch.zeros(t_horizon_exp)))

    assert torch.isclose(ego_trajectory_opt[0,  2], torch.zeros(1))
    assert torch.all(torch.isclose(ego_trajectory_opt[1:, 2], torch.ones(t_horizon_exp - 1)))
    assert torch.all(torch.eq(ego_trajectory_opt[:, 3], torch.zeros(t_horizon_exp)))

    time_steps_exp = torch.tensor([k * sim.dt for k in range(t_horizon_exp)])
    assert torch.all(torch.isclose(ego_trajectory_opt[:, -1], time_steps_exp))

    # Test ego planned trajectories - independent on environment engine. The ConstantSolver`s returned action
    # is always going straight with constant speed and direction, but just lasting for the first control step. Since
    # the ego is a single integrator agent, for the full remaining trajectory it stays at the point it went to
    # at the beginning (after the fist control loop).
    for k in range(t_horizon_exp - 1):
        assert torch.all(torch.isclose(ego_opt_planned[k, 1:, 0], torch.ones(solver.T) * (ego_trajectory_opt[k + 1, 0])))
        assert torch.all(torch.eq(ego_opt_planned[k, :, 1], torch.zeros(solver.T + 1)))
        time_steps_exp = torch.tensor([ego_trajectory_opt[k, -1] + i * sim.dt for i in range(solver.T + 1)])
        assert torch.all(torch.isclose(ego_opt_planned[k, :, -1], time_steps_exp))

    # Test ado planned trajectories - depending on environment engine. Therefore only time-stamps can be tested.
    for k in range(t_horizon_exp):
        time_steps_exp = torch.tensor([ego_trajectory_opt[k, -1] + i * sim.dt for i in range(solver.T + 1)])
        assert torch.all(torch.isclose(ego_opt_planned[k, :, -1], time_steps_exp))


def test_eval_environment():
    env = PotentialFieldEnvironment(IntegratorDTAgent, {"position": torch.tensor([-8, 0])})
    env.add_ado(position=torch.tensor([0, 0]), velocity=torch.tensor([-1, 0.2]))
    env.add_ado(position=torch.ones(2), velocity=torch.tensor([-5, 4.2]))

    # Since the determine ego controls are predictable in the constant solver, the output ego trajectory basically
    # is the unrolled trajectory by applying the same controls again and again.
    time_steps = 5
    ego_controls = torch.tensor([1, 0] * time_steps).view(-1, 2)
    ego_trajectory_opt_exp = env.ego.unroll_trajectory(controls=ego_controls, dt=env.dt)
    ado_trajectories_exp_sim = env.predict_w_controls(ego_controls=ego_controls)

    # First dont pass evaluation environment, then it planning and evaluation environment should be equal.
    # Ado Trajectories -> just the actual positions (first entry of planning trajectory at every time-step)
    solver = ConstantSolver(env, goal=torch.zeros(2), t_planning=2, objectives=[], constraints=[])
    ego_trajectory_opt, ado_trajectories = solver.solve(time_steps=time_steps)
    assert torch.all(torch.isclose(ego_trajectory_opt, ego_trajectory_opt_exp))
    assert torch.all(torch.isclose(ado_trajectories, ado_trajectories_exp_sim, atol=0.1))

    # Second pass evaluation environment. Since the environment updates should be performed using the evaluation
    # environment the ado trajectories should be equal to predicting the ado behaviour given the (known) ego controls.
    # To test multi-modality but still have a deterministic mode collapse all weight is put on one mode.
    # However, since the agents themselves are the same, especially the ego agent, the ego trajectory should be the
    # same as in the first test ==> (only one mode since deterministic)
    eval_env = SocialForcesEnvironment(IntegratorDTAgent, {"position": torch.tensor([-8, 0])})
    mode_kwargs = {"num_modes": 2, "weights": [1.0, 0.0]}
    eval_env.add_ado(position=torch.zeros(2), velocity=torch.tensor([-1, 0.2]), goal=torch.ones(2) * 10, **mode_kwargs)
    eval_env.add_ado(position=torch.ones(2), velocity=torch.tensor([-5, 4.2]), goal=torch.ones(2) * 10, **mode_kwargs)
    ado_trajectories_exp_eval = eval_env.predict_w_controls(ego_controls=ego_controls)[:, :1, :, :]  # first mode only
    solver = ConstantSolver(env, eval_env=eval_env, goal=torch.zeros(2), t_planning=2, objectives=[], constraints=[])
    ego_trajectory_opt, ado_trajectories = solver.solve(time_steps=time_steps)
    assert torch.all(torch.isclose(ego_trajectory_opt, ego_trajectory_opt_exp))
    assert torch.all(torch.isclose(ado_trajectories, ado_trajectories_exp_eval, atol=0.1))


###########################################################################
# Tests - All Solvers #####################################################
###########################################################################
# In order to test the general functionality of the solver base class, a simple optimization problem is
# created and solved, so that the feasibility of the result can be checked easily. This simple optimization problem
# basically is potential field (distance field) around a fixed point. In order to create this scenario an unconstrained
# optimization problem is posed with only a goal-based objective and no interaction with other agents. Then, all
# trajectory points should converge to the same point (which is the goal point).
@pytest.mark.parametrize(
    "solver_class", (MonteCarloTreeSearch, IGradSolver, SGradSolver)
)
class TestSolvers:

    @staticmethod
    def test_convergence(solver_class: Solver.__class__):
        sim = PotentialFieldEnvironment(IntegratorDTAgent, {"position": torch.tensor([-AGENT_SPEED_MAX / 2, 0])}, dt=1.0)
        solver = solver_class(sim, goal=torch.zeros(2), t_planning=1, objectives=[("goal", 1.0)], constraints=[])

        z0 = solver.z0s_default(just_one=True)
        z_opt, _, _ = solver.optimize(z0=z0, tag="core0", max_cpu_time=1.0, max_iter=1000)
        ego_controls = solver.z_to_ego_controls(z=z_opt.detach().numpy())
        ego_trajectory_opt = solver.env.ego.unroll_trajectory(controls=ego_controls, dt=solver.env.dt)

        assert torch.all(torch.isclose(ego_trajectory_opt[0, :], sim.ego.state_with_time))
        for k in range(1, solver.T):
            assert torch.all(torch.isclose(ego_trajectory_opt[k, 0:2], solver.goal, atol=0.5))

    @staticmethod
    def test_formulation(solver_class: Solver.__class__):
        sim, solver, z0 = scenario(solver_class)

        # Test output shapes.
        objective = solver.objective(z=z0, tag="core0")
        assert type(objective) == float
        constraints = solver.constraints(z=z0, tag="core0", return_violation=False)
        assert constraints.size == sum([c.num_constraints for c in solver.constraint_modules.values()])

    @staticmethod
    def test_x_to_x2(solver_class: IPOPTSolver.__class__):
        sim, solver, z0 = scenario(solver_class)
        x0 = solver.z_to_ego_trajectory(z0).detach().numpy()[:, 0:2]

        x02 = np.reshape(x0, (-1, 2))
        for xi in x02:
            assert any([np.isclose(np.linalg.norm(xk - xi), 0, atol=2.0) for xk in x0])

    @staticmethod
    def test_runtime(solver_class: IPOPTSolver.__class__):
        sim, solver, z0 = scenario(solver_class)

        comp_times_objective = []
        comp_times_constraints = []
        for _ in range(10):
            start_time = time.time()
            solver.constraints(z=z0, tag="core0")
            comp_times_constraints.append(time.time() - start_time)
            start_time = time.time()
            solver.objective(z=z0, tag="core0")
            comp_times_objective.append(time.time() - start_time)

        assert np.mean(comp_times_objective) < 0.04  # faster than 25 Hz (!)
        assert np.mean(comp_times_constraints) < 0.04  # faster than 25 Hz (!)


###########################################################################
# Test - IPOPT Solver #####################################################
###########################################################################
@pytest.mark.parametrize(
    "solver_class", (IGradSolver, SGradSolver)
)
class TestIPOPTSolvers:

    @staticmethod
    def test_formulation(solver_class: IPOPTSolver.__class__):
        sim, solver, z0 = scenario(solver_class)

        # Test output shapes.
        grad = solver.gradient(z=z0)
        assert np.linalg.norm(grad) > 0
        assert grad.size == z0.flatten().size
        jacobian = solver.jacobian(z0)
        num_constraints = sum([c.num_constraints for c in solver.constraint_modules.values()])
        assert jacobian.size == num_constraints * z0.flatten().size

    @staticmethod
    def test_runtime(solver_class: IPOPTSolver.__class__):
        sim, solver, z0 = scenario(solver_class)

        comp_times_jacobian = []
        comp_times_gradient = []
        for _ in range(10):
            start_time = time.time()
            solver.gradient(z=z0)
            comp_times_gradient.append(time.time() - start_time)
            start_time = time.time()
            solver.jacobian(z=z0)
            comp_times_jacobian.append(time.time() - start_time)
        assert np.mean(comp_times_jacobian) < 0.07  # faster than 15 Hz (!)
        assert np.mean(comp_times_gradient) < 0.07  # faster than 15 Hz (!)


def test_s_grad_solver():
    env = PotentialFieldEnvironment(IntegratorDTAgent, {"position": torch.tensor([-5, 0.5])})
    env.add_ado(position=torch.tensor([0, 0]), velocity=torch.tensor([-1, 0]))
    c_grad_solver = SGradSolver(env, goal=torch.tensor([5, 0]), t_planning=10, verbose=False)

    from mantrap.utility.primitives import square_primitives
    ego_path_init = square_primitives(start=env.ego.position, end=c_grad_solver.goal, dt=env.dt, steps=11)[1]
    ego_trajectory_init = env.ego.expand_trajectory(ego_path_init, dt=env.dt)
    ego_controls_init = env.ego.roll_trajectory(ego_trajectory_init, dt=env.dt)

    z_solution, _, _ = c_grad_solver.optimize(z0=ego_controls_init, tag="core0", max_cpu_time=5.0)
    ego_trajectory_opt = c_grad_solver.z_to_ego_trajectory(z=z_solution.detach().numpy())
    ado_trajectories = env.predict_w_trajectory(ego_trajectory=ego_trajectory_opt)

    for t in range(10):
        for m in range(env.num_ghosts):
            assert torch.norm(ego_trajectory_opt[t, 0:2] - ado_trajectories[m, :, t, 0:2]).item() >= CONSTRAINT_MIN_L2_DISTANCE


###########################################################################
# Test - ORCA Solver ######################################################
###########################################################################
class ORCAEnvironment(GraphBasedEnvironment):

    orca_rad = 1.0
    orca_dt = 10.0
    sim_dt = 0.25
    sim_speed_max = 4.0

    def __init__(self, ego_type=None, ego_kwargs=None, dt: float = sim_dt, **kwargs):
        super(ORCAEnvironment, self).__init__(ego_type, ego_kwargs, dt=self.sim_dt, **kwargs)
        self._ado_goals = []
        self._sim_time = self.time

    def add_ado(self, goal_position: Union[torch.Tensor, None] = torch.zeros(2), num_modes: int = 1, **ado_kwargs):
        assert num_modes == 1
        super(ORCAEnvironment, self).add_ado(ado_type=IntegratorDTAgent, log=False, num_modes=num_modes, **ado_kwargs)
        self._ado_goals.append(goal_position)

    def step(self, ego_policy: torch.Tensor = None) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        self._sim_time = self.time + self.dt

        assert self._ego is None, "environment merely should have ado agents"
        assert all([ghost.agent.__class__ == IntegratorDTAgent for ghost in self.ghosts])

        controls = torch.zeros((self.num_ados, 2))
        for ia, ghost in enumerate(self.ghosts):
            ado = ghost.agent  # uni-modal so ghosts = ados
            ado_kwargs = {"position": ado.position, "velocity": ado.velocity, "log": False}
            ado_env = ORCAEnvironment(IntegratorDTAgent, ego_kwargs=ado_kwargs)
            for other in self.ghosts[:ia] + self.ghosts[ia + 1:]:  # all agent except current loop element
                other_ado = other.agent  # uni-modal so ghosts = ados
                ado_env.add_ado(position=other_ado.position, velocity=other_ado.velocity, goal_position=None)

            ado_solver = ORCASolver(ado_env, goal=self._ado_goals[ia], t_planning=1, multiprocessing=False)
            controls[ia, :] = ado_solver.determine_ego_controls(
                speed_max=self.sim_speed_max, agent_radius=self.orca_rad, safe_dt=self.orca_dt
            )

        for j in range(self.num_ados):
            self._ado_ghosts[j].agent.update(controls[j, :], dt=self.dt)
        return torch.stack([ghost.agent.state for ghost in self.ghosts]), None

    def _build_connected_graph(self, t_horizon: int, ego_trajectory: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def build_connected_graph_wo_ego(self, t_horizon: int, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


def test_orca_single_agent():
    pos_init = torch.zeros(2)
    vel_init = torch.zeros(2)
    goal_pos = torch.ones(2) * 4

    pos_expected = torch.tensor([[0, 0], [0.70710, 0.70710], [1.4142, 1.4142], [2.1213, 2.1213], [2.8284, 2.8284]])

    sim = ORCAEnvironment()
    sim.add_ado(position=pos_init, velocity=vel_init, goal_position=goal_pos)

    assert sim.num_ados == 1
    assert torch.all(torch.eq(sim.ghosts[0].agent.position, pos_init))
    assert torch.all(torch.eq(sim.ghosts[0].agent.velocity, vel_init))

    pos = torch.zeros(pos_expected.shape)

    pos[0, :] = sim.ghosts[0].agent.position
    for k in range(1, pos.shape[0]):
        state_k, _ = sim.step()
        pos[k, :] = state_k[0, :2]

    assert torch.isclose(torch.norm(pos - pos_expected), torch.zeros(1), atol=0.1)


def test_orca_two_agents():
    pos_init = torch.tensor([[-5, 0.1], [5, -0.1]])
    vel_init = torch.zeros((2, 2))
    goal_pos = torch.tensor([[5, 0], [-5, 0]])

    pos_expected = torch.tensor(
        [
            [
                [-5, 0.1],
                [-4.8998, 0.107995],
                [-4.63883, 0.451667],
                [-3.65957, 0.568928],
                [-2.68357, 0.6858],
                [-1.7121, 0.802128],
                [-0.747214, 0.917669],
                [0.207704, 1.03202],
                [1.18529, 0.821493],
                [2.16288, 0.61097],
            ],
            [
                [5, -0.1],
                [4.8998, -0.107995],
                [4.63883, -0.451667],
                [3.65957, -0.568928],
                [2.68357, -0.6858],
                [1.7121, -0.802128],
                [0.747214, -0.917669],
                [-0.207704, -1.03202],
                [-1.18529, -0.821493],
                [-2.16288, -0.61097],
            ],
        ]
    )

    sim = ORCAEnvironment()
    sim.add_ado(position=pos_init[0, :], velocity=vel_init[0, :], goal_position=goal_pos[0, :])
    sim.add_ado(position=pos_init[1, :], velocity=vel_init[1, :], goal_position=goal_pos[1, :])

    pos = torch.zeros(pos_expected.shape)
    pos[0, 0, :] = sim.ghosts[0].agent.position
    pos[1, 0, :] = sim.ghosts[1].agent.position
    for k in range(1, pos.shape[1]):
        state_k, _ = sim.step()
        pos[:, k, :] = state_k[:, :2]

    assert torch.isclose(torch.norm(pos - pos_expected), torch.zeros(1), atol=0.1)
