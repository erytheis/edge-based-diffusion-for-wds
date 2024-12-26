import time
from typing import Any, Tuple
from typing import List, Optional

import numpy as np
import scipy
import torch
import torch_sparse
from hodgelaplacians import HodgeLaplacians
from line_profiler_pycharm import profile
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData
from torch_geometric.typing import OptTensor

from src.model.torch_models.data.data import GraphData, collate
from src.utils.torch_utils import sparse_eye


class SimplexData(GraphData):
    """
    Extends the torch_geometric.data.Data class to include a simplex attribute.
    At the moment is only limited to  2-simplex.
    """

    def __init__(self, x: OptTensor = None,
                 edge_index: OptTensor = None,
                 edge_attr: OptTensor = None,
                 y: OptTensor = None,
                 pos: OptTensor = None,
                 lower_laplacian_weight: OptTensor = None,
                 lower_laplacian_index: OptTensor = None,
1                 lower_boundary_weight: OptTensor = None,
                 lower_boundary_index: OptTensor = None,

                 **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)

        setattr(self._store, 'lower_laplacian_weight', lower_laplacian_weight)
        setattr(self._store, 'lower_laplacian_index', lower_laplacian_index)
        setattr(self._store, 'lower_boundary_weight', lower_boundary_weight)
        setattr(self._store, 'lower_boundary_index', lower_boundary_index)
        setattr(self._store, 'edge_y', kwargs.get('edge_y'))
        setattr(self._store, 'node_y', kwargs.get('node_y'))

    @property
    def lower_laplacian_weight(self):
        return self['lower_laplacian_weight'] if 'lower_laplacian_weight' in self._store else None

    @property
    def lower_laplacian_index(self):
        return self['lower_laplacian_index'] if 'lower_laplacian_index' in self._store else None

    @property
    def lower_boundary_weight(self):
        return self['lower_boundary_weight'] if 'lower_boundary_weight' in self._store else None

    @property
    def lower_boundary_index(self):
        return self['lower_boundary_index'] if 'lower_boundary_index' in self._store else None

    @property
    def edge_y(self):
        return self['edge_y']

    @property
    def node_y(self):
        return self['node_y']

    @profile
    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'batch' in key:
            return int(value.max()) + 1
        elif 'lower_laplacian_index' in key:
            return self.num_edges
        elif key == 'lower_boundary_index':
            boundary_inc = self.num_nodes
            cell_inc = self.num_edges
            inc = [[boundary_inc], [cell_inc]]
            return inc
        # elif 'boundary_index' in key:
        #     return self.num_edges
        elif 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0

    @classmethod
    def from_data_list(cls, data_list: List[BaseData],
                       follow_batch: Optional[List[str]] = None,
                       exclude_keys: Optional[List[str]] = None):
        r"""Constructs a :class:`~torch_geometric.data.Batch` object from a
        Python list of :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` objects.
        The assignment vector :obj:`batch` is created on the fly.
        In addition, creates assignment vectors for each key in
        :obj:`follow_batch`.
        Will exclude any keys given in :obj:`exclude_keys`."""

        batch, slice_dict, inc_dict = collate(
            Batch,
            data_list=data_list,
            increment=True,
            add_batch=not isinstance(data_list[0], Batch),
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
        )

        batch._num_graphs = len(data_list)
        batch._slice_dict = slice_dict
        batch._inc_dict = inc_dict

        return batch

    # def num_nodes(self) -> Optional[int]:
    #     return super().num_edges
    def mask_by_key(self, attribute, value, ):
        """
        Masks internal values in accordance with the
        Args:
            attribute: (str) attribute to return
            value: mask values that are NOT equal to this parameter
        :
        """
        slices = self._store._slice_dict
        mask = np.array(self.wds_names) == value
        start_idx = slices[attribute][:-1][mask]
        end_idx = slices[attribute][1:][mask]
        mask = torch.zeros_like(self[attribute], dtype=torch.bool)
        for l, r in zip(start_idx, end_idx):
            mask[l:r] = 1
        return mask

    def mask_by_features(self, attribute='edge_attr', column_idx=-1, value=0):
        """
        Masks internal values in accordance with the
        Args:
            attribute: (str) attribute to return
            column_idx: (int) column index of the feature to mask
        :
        """
        mask = self[attribute][:, column_idx] == value
        return mask


    def get_adjacency(self, dim: Tuple[int]):
        """
        Get adjacency by given dimensions.
        :param dim:
        :return:

        Example:
        1:
            >>> data = SimplexData(edge_index=torch.tensor([[0, 1, 1, 2],
            ...                                             [1, 0, 2, 1]]))

            >>> data.get_adjacency((0, 0))
            torch.tensor([[0, 1, 1, 2],
            [1, 0, 2, 1]]
        2:
            >>> data = SimplexData(lower_laplacian_index=torch.tensor([[0, 1, 1, 2],
            ...                                                   [1, 0, 2, 1]]))

            >>> data.get_adjacency((1, 1))
            torch.tensor([[0, 1, 1, 2],
            [1, 0, 2, 1]]
        """
        if dim == (0, 0):
            return self['edge_index']
        elif dim == (1, 1):
            return self['lower_laplacian_index']
        elif dim == (0, 1):
            return self['boundary_index']
        elif dim == (1, 0):
            return self['boundary_index'].flip(0)
        else:
            raise ValueError(f'Invalid dim: {dim}')

    def get_weights(self, dim: Tuple[int]):
        """
        Get weights by given dimensions.
        :param dim:
        :return:

        Example:
        1:
            >>> data = SimplexData(edge_weight=torch.tensor([1, 2, 3, 4]))

            >>> data.get_weights((0, 0))
            torch.tensor([1, 2, 3, 4]
        2:
            >>> data = SimplexData(lower_laplacian_weight=torch.tensor([1, 2, 3, 4]))

            >>> data.get_weights((1, 1))
            torch.tensor([1, 2, 3, 4]
        """
        if dim == (0, 0):
            return self['edge_weight']
        elif dim == (1, 1):
            return self['lower_laplacian_weight']
        elif dim == (0, 1):
            return self['boundary_weight']
        elif dim == (1, 0):
            return self['boundary_weight']
        else:
            raise ValueError(f'Invalid dim: {dim}')



def get_lower_boundary(data: SimplexData, device='cpu', weight_idx=None):
    ei = data.edge_index

    boundary_src = ei.reshape(1, -1).squeeze()
    boundary_dst = torch.arange(0, ei.shape[1], device=device, dtype=torch.long).repeat(2)
    boundary_index = torch.stack((boundary_src, boundary_dst))

    # add weights
    if weight_idx is None:
        boundary_weight = torch.ones(ei.shape[1] * 2, dtype=torch.float, device=device)
        boundary_weight[:ei.shape[1]] = -1
    else:
        boundary_weight = data.edge_attr[:, weight_idx].abs().repeat(2) ** 0.5
        boundary_weight[:ei.shape[1]] *= -1
    return boundary_index, boundary_weight


@profile
def get_lower_boundary_and_laplacian(data: SimplexData, normalized=True, remove_self_loops=False,
                                     device='cpu',
                                     release_ends_of_virtual_edges=False,
                                     weight_idx=None):
    boundary_index, boundary_weight = get_lower_boundary(data, device, weight_idx)
    #
    B1 = torch.sparse_coo_tensor(boundary_index, boundary_weight, size=(data.num_nodes, data.num_edges))
    # B1 = B1.coalesce()

    L = get_L_first_option(B1, normalized)
    if release_ends_of_virtual_edges:
        virtual_edges = data.edge_attr[:, -1] == 1
        L = (L - sparse_eye(L.shape[0],
                            bool_vector=virtual_edges,
                            device=device))

    if remove_self_loops:
        eye = sparse_eye(L.shape[0], value=2., device=device)
        L = L - eye.coalesce()

    lower_laplacian_index = L.coalesce().indices()
    lower_laplacian_weight = L.coalesce().values()

    return boundary_index, boundary_weight, lower_laplacian_index, lower_laplacian_weight


@profile
def get_L_first_option(B1, normalized=True, ):
    B1 = B1.to_dense() if hasattr(B1, 'to_dense') else B1

    if normalized:
        D = torch.diag(1 / torch.sum(torch.abs(B1), dim=1))
        B_norm = B1.T @ D
        L = torch.mm(B_norm, B1)
    else:
        L = torch.sparse.mm(B1.T, B1)

    L = L.to_sparse()
    return L


@profile
def get_L_torch_sparse(boundary_index, boundary_weight, data, B1=None):
    if B1 is None:
        B1 = torch.sparse_coo_tensor(boundary_index, boundary_weight, size=(data.num_nodes, data.num_edges))
        B1 = B1.coalesce()

    index_T, values_T = torch_sparse.transpose(boundary_index, boundary_weight, data.num_nodes, data.num_edges)
    index_T, values_T = torch_sparse.coalesce(index_T, values_T, data.num_edges, data.num_nodes, op="add")

    D = torch.sparse.sum(torch.abs(B1), dim=1)
    diagonals_values = torch.reciprocal(D.values())
    diagonals_index = torch.arange(diagonals_values.shape[0], device=D.device).repeat(2, 1)
    diagonals_index, diagonals_values = torch_sparse.coalesce(diagonals_index, diagonals_values,
                                                              diagonals_values.shape[0], diagonals_values.shape[0],
                                                              op="add")

    L_indices, L_values = torch_sparse.spspmm(index_T, values_T, diagonals_index, diagonals_values,
                                              data.num_edges, data.num_nodes, data.num_nodes)

    L = torch_sparse.spspmm(L_indices, L_values, boundary_index, boundary_weight, data.num_edges, data.num_nodes,
                            data.num_edges)
    return L


