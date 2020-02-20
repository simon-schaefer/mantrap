import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.simulation import SocialForcesSimulation
from mantrap.solver import IGradSolver

ego_position = torch.tensor([-7, 0])
ego_velocity = torch.zeros(2)
ego_goal = torch.tensor([7, 0])
ado_positions = torch.stack((torch.tensor([-7, -4]), torch.tensor([7, 6]), torch.tensor([7, -3])))
ado_goals = torch.stack((torch.tensor([0, 0]), torch.tensor([-7, 0]), torch.tensor([-7, 4])))
ado_velocities = torch.stack((torch.tensor([1, 0]), torch.tensor([-1, 0]), torch.tensor([-1, 1])))

sim = SocialForcesSimulation(IntegratorDTAgent, {"position": ego_position, "velocity": ego_velocity})
# for position, goal, velocity in zip(ado_positions, ado_goals, ado_velocities):
#     sim.add_ado(position=position, goal=goal, velocity=velocity, num_modes=2)
sim.add_ado(position=torch.tensor([-7, -4]), goal=torch.tensor([0, 0]), velocity=torch.tensor([1, 0]), num_modes=1)
solver = IGradSolver(sim, goal=ego_goal, verbose=True, T=10)

x_opt, ado_states = solver.solve(max_iter=100, max_cpu_time=10.0)

print(x_opt)
print(ado_states)
