import numpy as np
from line_profiler_pycharm import profile

import src.model
from src.model.torch_models.data.complex import get_upper_boundary_and_laplacian, ComplexData, \
    get_upper_boundary_and_laplacian_sparse
from src.model.torch_models.data.simplex import get_lower_boundary_and_laplacian, SimplexData
from src.model.torch_models.dataset.transforms.base import BaseTransform


class ToData(BaseTransform):

    def forward(self, data, *args, **kwargs):
        x = data.x
        num_graphs = 1
        y = data.y.view(-1, x.shape[0] * num_graphs)
        x = x.view(-1, x.shape[1] * x.shape[0] * num_graphs)
        return src.model.torch_models.data.data.Data(x=x, y=y,
                                                                node_names=self.node_names,
                                                                x_names=self.x_names,
                                                                y_names=self.y_names,
                                                                wds_names=self.wds_names
                                                                )

    def _infer_parameters(self, data, *args, **kwargs):
        self.node_names = data.node_names,
        self.x_names = data.x_names,
        self.y_names = data.y_names,
        self.wds_names = data.wds_names


class ToSimplexData(BaseTransform):

    def __init__(self, edge_label='flowrate', node_label=None, normalized=False,
                 remove_self_loops=False, iterative_smoothing_coefficient=0.5,
                 release_ends_of_virtual_edges=False,
                 **kwargs):
        self.edge_label = edge_label
        self.node_label = node_label
        self.normalized = normalized
        self.remove_self_loops = remove_self_loops
        self.release_ends_of_virtual_edges = release_ends_of_virtual_edges
        self.init_kwargs = kwargs
        self.iterative_smoothing_coefficient = iterative_smoothing_coefficient
        super().__init__()

    @profile
    def forward(self, data, *args, **kwargs):
        B1_i, B1_w, L_down_i, L_down_w = get_lower_boundary_and_laplacian(data,
                                                              self.normalized,
                                                              self.remove_self_loops,
                                                              release_ends_of_virtual_edges =self.release_ends_of_virtual_edges,
                                                              iterative_smoothing_coefficient=self.iterative_smoothing_coefficient,
                                                              weight_idx=None,
                                                              device=data.x.device)


        if self.edge_label is not None:
            kwargs['edge_y'] = data.edge_attr[:, self.edge_label_index].clone()
            y = kwargs['edge_y']

        if self.node_label is not None:
            kwargs['node_y'] = data.x[:, self.node_label_index].clone()
        else:
            kwargs['node_y'] = data.y.clone()

        return SimplexData(x=data.x,
                           y=data.y,
                           edge_attr=data.edge_attr,
                           edge_index=data.edge_index,
                           wds_names=data.wds_names,
                           lower_laplacian_index=L_down_i,
                           lower_laplacian_weight=L_down_w,
                           lower_boundary_index=B1_i,
                           lower_boundary_weight=B1_w,
                           **self.init_kwargs,
                           **kwargs,
                           )

    def _infer_parameters(self, data, *args, **kwargs):
        self.edge_label_index = data.edge_attr_names.index(self.edge_label)
        if self.node_label is not None:
            self.node_label_index = data.x_names.index(self.node_label)


class ToComplexData(BaseTransform):

    def __init__(self, edge_label='flowrate', node_label=None, normalized=False,
                 remove_self_loops=False, iterative_smoothing_coefficient_down=0.0,
                 iterative_smoothing_coefficient_up=0.0, **kwargs):
        self.edge_label = edge_label
        self.node_label = node_label
        self.normalized = normalized
        self.remove_self_loops = remove_self_loops
        self.init_kwargs = kwargs

        super().__init__()

    def forward(self, data, *args, **kwargs):
        B1_i, B1_w, L_down_i, L_down_w = get_lower_boundary_and_laplacian(data,
                                                              self.normalized,
                                                              self.remove_self_loops,
                                                              weight_idx=None,
                                                              device=data.x.device)

        L1_up_i, L1_up_w, B2_i, B2_w = get_upper_boundary_and_laplacian_sparse(data, device=data.x.device,
                                                                        )

        if self.edge_label is not None:
            kwargs['edge_y'] = data.edge_attr[:, self.edge_label_index].clone()
            y = kwargs['edge_y']

        if self.node_label is not None:
            kwargs['node_y'] = data.x[:, self.node_label_index].clone()
        else:
            kwargs['node_y'] = data.y.clone()

        return ComplexData(x=data.x,
                           y=data.y,
                           edge_attr=data.edge_attr,
                           edge_index=data.edge_index,
                           wds_names=data.wds_names,
                           # lower
                           lower_laplacian_index=L_down_i,
                           lower_laplacian_weight=L_down_w,
                           lower_boundary_index=B1_i,
                           lower_boundary_weight=B1_w,
                           # upper
                           upper_laplacian_index=L1_up_i,
                           upper_laplacian_weight=L1_up_w,
                           upper_boundary_index=B2_i,
                           upper_boundary_weight=B2_w,
                           **self.init_kwargs,
                           **kwargs,
                           )

    def _infer_parameters(self, data, *args, **kwargs):
        self.edge_label_index = data.edge_attr_names.index(self.edge_label)
        if self.node_label is not None:
            self.node_label_index = data.x_names.index(self.node_label)
