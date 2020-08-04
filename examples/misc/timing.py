import mantrap
import torch


env = mantrap.environment.Trajectron(ego_position=torch.tensor([0.0, -1.0]))
env.add_ado(position=torch.tensor([3, 2]), velocity=torch.tensor([0, -1.0]))
env.add_ado(position=torch.tensor([-3, 7]), velocity=torch.tensor([1.0, -1.0]))
env.add_ado(position=torch.tensor([-4, -3]), velocity=torch.tensor([1.0, 1.0]))

solver = mantrap.solver.IPOPTSolver(env=env, goal=torch.tensor([8, 0]))
_, _,  = solver.solve(time_steps=30)
