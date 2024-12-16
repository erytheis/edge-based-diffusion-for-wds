import time
from itertools import zip_longest

import networkx as nx
import numpy as np
import scipy as sp
import torch
from line_profiler_pycharm import profile
from torch_geometric.typing import OptTensor

from src.surrogate_models.torch_models.data.simplex import SimplexData


class ComplexData(SimplexData):
    """
    Extends the torch_geometric.data.Data class to include a simplex attribute.
    At the moment is only limited to  2-simplex.
    """

    def __init__(self, x: OptTensor = None,
                 edge_index: OptTensor = None,
                 edge_attr: OptTensor = None,
                 y: OptTensor = None,
                 pos: OptTensor = None,
                 up_laplacian_weight: OptTensor = None,
                 up_laplacian_index: OptTensor = None,
                 **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)

        setattr(self._store, 'up_laplacian_weight', up_laplacian_weight)
        setattr(self._store, 'up_laplacian_index', up_laplacian_index)

    @property
    def laplacian_index_up(self):
        return self['up_laplacian_index'] if 'up_laplacian_index' in self._store else None

    @property
    def laplacian_weight_up(self):
        return self['up_laplacian_weight'] if 'up_laplacian_weight' in self._store else None

@profile
def get_L_up(data, normalized=False):
    edges = list(data.edge_index.t().cpu().numpy())

    st_time = time.time()
    graph = nx.from_edgelist(data.edge_index.t().cpu().numpy())
    # # if normalized:
    # cycles = nx.minimum_cycle_basis(graph)
    # for i, cycle in enumerate(cycles):
    #     cycle = list(np.array(nx.find_cycle(graph.subgraph(cycle)))[:, 0])
    #     cycles[i] = cycle
    # #
    cycles = nx.cycle_basis(nx.from_edgelist(data.edge_index.t().cpu().numpy()))

    end_time = time.time()

    print('time:', end_time - st_time)
    # cycles = nx.minimum_cycle_basis(nx.from_edgelist(data.edge_index.t().cpu().numpy()))
    # cycles = nx.chordless_cycles(nx.from_edgelist(data.edge_index.t().cpu().numpy()))

    edgelist = [e for e in edges]

    A = sp.sparse.lil_matrix((len(edgelist), len(cycles)))

    edge_index = {
        tuple(edge): i for i, edge in enumerate(edgelist)
    }  # orient edges

    for celli, cell in enumerate(cycles):
        edge_visiting_dic = {}  # this dictionary is cell dependent
        # mainly used to handle the cell complex non-regular case
        for edge in list(
                zip_longest(cell, cell[1:] + [cell[0]])
        ):
            ei = edge_index[tuple(sorted(edge))]
            if ei not in edge_visiting_dic:
                if edge in edge_index:
                    edge_visiting_dic[ei] = 1
                else:
                    edge_visiting_dic[ei] = -1
            else:
                if edge in edge_index:
                    edge_visiting_dic[ei] = edge_visiting_dic[ei] + 1
                else:
                    edge_visiting_dic[ei] = edge_visiting_dic[ei] - 1

            A[ei, celli] = edge_visiting_dic[
                ei
            ]
    B2 = A

    if normalized:
        # add diagonal matrix with the same size as the number of cycles and length of cycles as entry
        D3 = np.diag([1 / len(c) for c in cycles])
        D2 = np.abs(B2).sum(axis=1).squeeze()
        D2 = np.clip(D2, 1, None)
        D2 = sp.sparse.spdiags(1 / D2, 0, D2.size, D2.size)

        L1_up = (B2 @ D3 @ B2.T @ D2).astype(np.float32)
        # L1_up = (B2 @ D3 @ B2.T).astype(np.float32)
    else:
        L1_up = (B2 @ B2.T).astype(np.float32)

    return L1_up, B2

def get_L_up_planar(data, normalized=False):
    edges = list(data.edge_index.t().cpu().numpy())

    graph = nx.from_edgelist(data.edge_index.t().cpu().numpy())
    # # if normalized:
    faces = get_faces(graph)

    faces = list(faces)

    # remove face with the largest number of edges
    face_lengths = [len(f) for f in faces]
    faces.pop(np.argmax(face_lengths))

    for i, cycle in enumerate(faces):
        cycle = list(nx.find_cycle(nx.from_edgelist(cycle)))
        faces[i] = cycle
    edgelist = [e for e in edges]

    A = sp.sparse.lil_matrix((len(edgelist), len(faces)))

    edge_index = {
        tuple(edge): i for i, edge in enumerate(edgelist)
    }  # orient edges

    for celli, cell in enumerate(faces):
        edge_visiting_dic = {}  # this dictionary is cell dependent
        # mainly used to handle the cell complex non-regular case

        for edge in cell:
            ei = edge_index[tuple(sorted(edge))]
            if ei not in edge_visiting_dic:
                if edge in edge_index:
                    edge_visiting_dic[ei] = 1
                else:
                    edge_visiting_dic[ei] = -1
            else:
                if edge in edge_index:
                    edge_visiting_dic[ei] = edge_visiting_dic[ei] + 1
                else:
                    edge_visiting_dic[ei] = edge_visiting_dic[ei] - 1

            A[ei, celli] = edge_visiting_dic[
                ei
            ]
    B2 = A
    if normalized:
        # add diagonal matrix with the same size as the number of cycles and length of cycles as entry
        D3 = np.diag([1 / len(c) for c in faces])
        D2 = np.abs(B2).sum(axis=1).squeeze()
        D2 = np.clip(np.sqrt(D2), 1, None)
        D2 = sp.sparse.spdiags(1 / D2, 0, D2.size, D2.size)

        L1_up = (D2 @ B2 @ D3 @ B2.T @ D2).astype(np.float32)

    else:
        L1_up = (B2 @ B2.T).astype(np.float32)


    return  L1_up, B2


def get_L_up_sparse(data, device, iterative_smoothing_coefficient=0.0):
    L1_up,_ = get_L_up(data)

    if iterative_smoothing_coefficient is not None and iterative_smoothing_coefficient > 0.0:
        L1_up= sp.sparse.identity(L1_up.shape[0], format='csr')- L1_up * iterative_smoothing_coefficient

    L1_up = L1_up.tocoo()

    values = L1_up.data
    indices = np.vstack((L1_up.row, L1_up.col))

    i = torch.tensor(indices, dtype=torch.long, device=device)
    v = torch.tensor(values, dtype=torch.float, device=device)

    return i, v


def get_faces(graph):
    is_planar, embedding = nx.check_planarity(graph)
    if not is_planar:
        raise Exception("Graph cannot be made planar")

    # now, initialize an empty set of faces
    faces = set()
    # and a face->geometry dictionary to store the
    # geometry that corresponds to a specifc face
    geometries = dict()
    # then, for each node
    for i in graph.nodes:
        for j in graph[i]:  # and its neighbors
            # get the face from the embedding that corresponds to that edge
            face = embedding.traverse_face(i, j)
            # then, build the "path" of edges from start to finish
            path = list(zip(face, face[1:] + [face[0]]))
            # construct a canonical ordering for the edges in the face
            # i.e. the face built from ((2,1),(1,0),(0,2))
            #      and ((1,0), (0,2), (2,1)) should be the same
            face_tuple = tuple(sorted(tuple(sorted(x)) for x in path))
            # then, add this face with sorted edges to the final set of faces
            faces.add(face_tuple)
            # note that, since it's a set, its uniqueness is guaranteed

    return faces