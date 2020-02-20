import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.simulation import SocialForcesSimulation
from mantrap.solver import IGradSolver
from mantrap.evaluation.visualization import visualize_scenes
from mantrap.utility.io import build_os_path

ego_position = torch.tensor([-7, 0])
ego_velocity = torch.zeros(2)
ego_goal = torch.tensor([7, 0])
ado_positions = torch.stack((torch.tensor([-7, -1]), torch.tensor([7, 3]), torch.tensor([7, -2])))
ado_goals = torch.stack((torch.tensor([0, 0]), torch.tensor([-7, 0]), torch.tensor([-7, 4])))
ado_velocities = torch.stack((torch.tensor([1, 0]), torch.tensor([-1, 0]), torch.tensor([-1, 1])))

sim = SocialForcesSimulation(IntegratorDTAgent, {"position": ego_position, "velocity": ego_velocity})
for position, goal, velocity in zip(ado_positions, ado_goals, ado_velocities):
    sim.add_ado(position=position, goal=goal, velocity=velocity, num_modes=2)
solver = IGradSolver(sim, goal=ego_goal, verbose=False, T=10, num_control_points=2)

x_opt, ado_states, x_opt_planned = solver.solve(horizon=20, max_iter=10, max_cpu_time=2.0)
visualize_scenes(x_opt_planned, ado_states, env=sim, file_path=build_os_path("test/graphs/sim_exp"))
