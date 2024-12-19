import torch_geometric
from torch_geometric.typing import OptTensor, Tensor, Adj

import torch


class SimplicialFeaturePropagation(torch_geometric.nn.conv.GCNConv):
    def __init__(self, *args, **kwargs, ):
        super().__init__(in_channels=1, out_channels=1, *args, **kwargs, normalize=False)
        self.lin = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class DiffusionStep(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SimplicialFeaturePropagation(node_dim=0, flow='target_to_source')

    def forward(self, x, edge_index,
                edge_weight):
        return self.conv1(x, edge_index,
                          edge_weight)
