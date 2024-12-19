import torch
from torch_geometric.graphgym import GCNConv
from torch_geometric.typing import OptTensor, Tensor, Adj


class SimplicialFeaturePropagation(GCNConv):
    def __init__(self, *args, **kwargs, ):
        super().__init__(in_channels=1, out_channels=1, *args, **kwargs, normalize=False)
        self.lin = None

    @profile
    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j




def get_heads_from_flowrates_linalg(batch, output, dH_predicted=None,
                                    reservoir_idx=3,
                                    junction_idx=2,
                                    edge_mask=None):
    if dH_predicted is None:
        loss_coefficient = batch.edge_attr[:, 1]
        dH_predicted = torch.abs(output) ** 1.852 * loss_coefficient * torch.sign(output)

    B1 = torch.sparse_coo_tensor(batch.boundary_index, batch.boundary_weight,
                                 size=(batch.num_nodes, batch.num_edges))
    B1_ = B1.to_dense()

    # remove virtual_edges
    B1_ = B1_[:, edge_mask]

    # add known heads
    known_nodes = batch.x[:, reservoir_idx] == 1
    unknown_nodes = batch.x[:, junction_idx] == 1
    B1_known = B1_[known_nodes].T
    known_heads = batch.node_y[known_nodes].unsqueeze(-1)
    # known_heads = batch.x[known_nodes, 1].unsqueeze(-1)

    dH_known = (B1_known @ known_heads).squeeze()

    # get prediction
    solution = torch.zeros(batch.num_nodes, device=batch.x.device)
    solution[known_nodes] = known_heads.squeeze()
    #
    solution[unknown_nodes] = torch.linalg.lstsq(B1_[unknown_nodes].T, dH_predicted - dH_known,
                                                 driver='gels',
                                                 ).solution.squeeze()

    return solution


def from_flowrates_to_heads(batch, output_, node_sensor_idx=3, **kwargs):
    output_['x'] = torch.zeros_like(batch.node_y.unsqueeze(-1))

    for i in range(len(batch)):
        batch_ = batch[i]

        # get positioning
        edge_mask = batch_.mask_by_features(value=0, column_idx=2)
        # node_mask = batch_.mask_by_features(value=0, column_idx=4, attribute='x')

        slices_x = slice(batch.slices['x'][i], batch.slices['x'][i + 1])
        slices_edge_attr = slice(batch.slices['edge_attr'][i], batch.slices['edge_attr'][i + 1])

        # remove virtual edges
        output = output_['edge_attr'][slices_edge_attr]
        output = output[edge_mask].squeeze()
        # batch_.drop_node_by_features(nodes_value=0, edges_value=0, edges_column_idx=2)

        # get edge prediction
        loss_coefficient = batch_.edge_attr[:, 1][edge_mask]
        dH_predicted = torch.abs(output) ** 1.852 * loss_coefficient * torch.sign(output)

        # get true values
        heads = batch_.x[:, 1]

        # derive node-prediction
        node_prediction = get_heads_from_flowrates_linalg(batch_, output, dH_predicted, node_sensor_idx,
                                                          edge_mask=edge_mask)

        output_['x'][slices_x] = node_prediction.unsqueeze(-1)

    return output_
