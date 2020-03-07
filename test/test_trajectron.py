import torch

from mantrap.agents import DoubleIntegratorDTAgent
from mantrap.simulation import Trajectron

if __name__ == '__main__':
    trajectron = Trajectron(DoubleIntegratorDTAgent, {"position": torch.zeros(2)})
    trajectron.add_ado(
        position=torch.tensor([1, 0]),
        velocity=torch.tensor([0, 0]),
        history=torch.stack(5 * [torch.tensor([1, 0, 0, 0, 0])])
    )
    trajectron.add_ado(
        position=torch.tensor([0, 1]),
        velocity=torch.tensor([0, 0]),
        history=torch.stack(5 * [torch.tensor([0, 1, 0, 0, 0])])
    )
    a = trajectron.predict_wo_ego(t_horizon=10)

    ego_trajectory = torch.rand((5, 6), requires_grad=True)
    b = trajectron.predict_w_trajectory(trajectory=ego_trajectory)

    for node, node_gmm in b.items():
        print(node, node_gmm.mus.grad_fn, node_gmm.mus.shape)
        grad = torch.autograd.grad(node_gmm.mus[0, 0, 0, 0, 0], ego_trajectory, retain_graph=True)

