from datetime import time

import torch
from torch import nn

from src.model.torch_models.model.diffusion_step import DiffusionStep
from src.model.torch_models.model.divergence_checker import DivergenceChecker
from src.model.torch_models.model.metric import r2_score


class Diffusion(nn.module):
    def __init__(self, convergence_crierion):

        self.fp = DiffusionStep()
        self.convergence_criterion = convergence_crierion
        self.divergence_checker = DivergenceChecker


    def forward(self, batch, mu_down, mu_up, momentum_up, momentum_down, k=3):

        #TODO make these into batch an attribute
        num_real_nodes = batch.num_nodes // 2
        num_real_nodes = int(num_real_nodes / len(batch))

        virtual = (batch.edge_attr[:, -2] == 1).unsqueze(-1)
        reservoir_connector = (batch.edge_attr[:, -1] == 1).unsqueeze(-1)


        # account for multiple reservoirs
        known_nodes = batch.x[:, 3] == 1

        if known_nodes.sum() > len(batch):
            virtual_reservoir_nodes = torch.zeros_like(batch.x[:, 3], dtype=torch.bool)

            edge_attr_slices = batch.slices['edge_attr'].numpy()
            for i in range(len(edge_attr_slices) - 1):
                x_slices = batch.slices['x'].int().numpy()

                virtual_reservoir_nodes[(x_slices[i] + num_real_nodes):x_slices[i + 1]] = \
                    (batch.x[:, 3] == 1)[x_slices[i]:(x_slices[i + 1] - num_real_nodes)]

            virtual_reservoir_pipes = virtual_reservoir_nodes[batch.edge_index[0]] | \
                                      virtual_reservoir_nodes[batch.edge_index[1]]

            # release the end of the network
            # sparse for FP
            L1_down_sparse_indices, L1_down_sparse_values = batch.lower_laplacian_index, batch.lower_laplacian_weight

            virtual_reservoir_pipes_sparse_indices = virtual_reservoir_pipes[L1_down_sparse_indices[0]] & \
                                                     virtual_reservoir_pipes[
                                                         L1_down_sparse_indices[1]]

            virtual[virtual_reservoir_pipes] = 0

            L1_down_sparse_values[virtual_reservoir_pipes_sparse_indices] = 1
            batch.lower_laplacian_index = L1_down_sparse_indices
            batch.lower_laplacian_weight = L1_down_sparse_values

            h_known = reservoir_connector * h_real

        f_true = batch.edge_y.unsqueeze(-1)

        # remove reservoir from mask
        mask = virtual.squeeze()

        loss_coefficient = batch.edge_attr[:, 1].unsqueeze(-1)
        loss_coefficient = torch.clip(loss_coefficient, 10e-6)

        f = f_true.clone()
        f[~virtual] = 0

        # scatter virtual edges to sparse indices
        mask_ = virtual[batch.upper_laplacian_index[0]].squeeze()

        index_up = batch.upper_laplacian_index[:, ~mask_]
        weight_up = batch.upper_laplacian_weight[~mask_].clone() * mu_up

        index_down = batch.lower_laplacian_index
        weight_down = batch.lower_laplacian_weight.clone() * mu_down

        for i in self.convergence_criterion:
            # lower gradient steps
            for _ in range(k):
                f, delta_f = self._downward_step(
                    f, delta_f, index_down, weight_down,
                    momentum_down, virtual, f_true
                )

            # Transform to h
            h = self._transform_to_h(f, loss_coefficient)

            # upper_gradient steps
            h, delta_h = self._upward_step(
                h, delta_h, index_up, weight_up,
                momentum_up, known_nodes, reservoir_connector, h_known, batch
            )

            f = self._transform_to_y(h, f, loss_coefficient, reservoir_connector)

            self.convergence_criterion.update(f, h)

    def _downward_step(self, y_mask, delta_y_mask, index, weight, momentum, virtual, y):
        y_mask_old = y_mask
        y_mask = y_mask - self.fp(y_mask, index, weight) + momentum * delta_y_mask
        # Apply virtual constraints
        y_mask = self._apply_constraints(y_mask, virtual, y)
        delta_y_mask = y_mask - y_mask_old
        return y_mask, delta_y_mask

    def _upward_step(self, h, delta_h, index, weight, momentum, known_nodes, reservoir_connector, h_known, batch):
        # Enforce known nodes if condition met
        if known_nodes.sum() > len(batch):
            h = torch.where(reservoir_connector, h_known, h)

        h_old = h
        h = h - self.fp(h, index, weight) + momentum * delta_h
        delta_h = h - h_old
        return h, delta_h

    def _transform_to_h(self, y_mask, loss_coefficient):
        # h = loss_coefficient * |y_mask|^1.852 * sign(y_mask)
        return loss_coefficient * y_mask.abs().pow(1.852) * y_mask.sign()

    def _transform_to_f(self, h, y_mask, loss_coefficient, reservoir_connector):
        # y_new = ((|h| / loss_coefficient)^(1/1.852)) * sign(h)
        y_new = (h.abs() / loss_coefficient).pow(1 / 1.852) * h.sign()
        # For reservoir connectors, keep old y_mask
        y_mask = torch.where(~reservoir_connector, y_new, y_mask)
        return y_mask




