import numpy as np
import torch

from torch import nn

from src.model.torch_models.model.convergence.convergence_checker import (
    StopPointsConvergenceChecker,
)
from src.model.torch_models.model.diffusion_step import DiffusionStep


class Diffusion(nn.Module):
    def __init__(self, convergence_checker=None):

        super().__init__()
        self.fp = DiffusionStep()

        if convergence_checker is None:
            convergence_checker = StopPointsConvergenceChecker(
                np.linspace(0, 1000, 100).astype(int)
            )

        self.convergence_checker = convergence_checker

    #@profile
    def forward(self, batch, tau_down, tau_up, momentum_down, momentum_up, k=3):

        device = batch.edge_attr.device

        # TODO make these into batch an attribute
        num_real_nodes = batch.num_nodes // 2
        num_real_nodes = int(num_real_nodes / len(batch))

        virtual = (batch.edge_attr[:, -2] == 1).unsqueeze(-1)
        virtual_nodes = (batch.x[:, -2] == 1).unsqueeze(-1)
        reservoir_connector = (batch.edge_attr[:, -1] == 1).unsqueeze(-1)

        # account for multiple reservoirs
        known_nodes = batch.x[:, 3] == 1

        if known_nodes.sum() > len(batch):
            virtual_reservoir_nodes = torch.zeros_like(batch.x[:, 3], dtype=torch.bool)

            edge_attr_slices = batch.slices["edge_attr"].numpy()
            for i in range(len(edge_attr_slices) - 1):
                x_slices = batch.slices["x"].int().numpy()

                virtual_reservoir_nodes[
                    (x_slices[i] + num_real_nodes) : x_slices[i + 1]
                ] = (batch.x[:, 3] == 1)[
                    x_slices[i] : (x_slices[i + 1] - num_real_nodes)
                ]

            virtual_reservoir_pipes = (
                virtual_reservoir_nodes[batch.edge_index[0]]
                | virtual_reservoir_nodes[batch.edge_index[1]]
            )

            # release the end of the network
            # sparse for FP
            L1_down_sparse_indices, L1_down_sparse_values = (
                batch.lower_laplacian_index,
                batch.lower_laplacian_weight,
            )

            virtual_reservoir_pipes_sparse_indices = (
                virtual_reservoir_pipes[L1_down_sparse_indices[0]]
                & virtual_reservoir_pipes[L1_down_sparse_indices[1]]
            )

            virtual[virtual_reservoir_pipes] = 0

            L1_down_sparse_values[virtual_reservoir_pipes_sparse_indices] = 1
            batch.lower_laplacian_index = L1_down_sparse_indices
            batch.lower_laplacian_weight = L1_down_sparse_values

            h_known = reservoir_connector * h_real
        else:
            h_known = None

        f_true = batch.edge_attr[:,0].unsqueeze(-1)

        # remove reservoir from mask
        mask = virtual.squeeze()

        loss_coefficient = batch.edge_attr[:, 1].unsqueeze(-1)
        loss_coefficient = torch.clip(loss_coefficient, 10e-6)

        f = f_true.clone()
        f[~virtual] = 0

        delta_f = torch.zeros_like(f, device=device)
        delta_h = torch.zeros_like(f, device=device)

        # scatter virtual edges to sparse indices
        mask_ = virtual[batch.upper_laplacian_index[0]].squeeze()

        index_up = batch.upper_laplacian_index[:, ~mask_]
        weight_up = batch.upper_laplacian_weight[~mask_].clone() * tau_up

        index_down = batch.lower_laplacian_index
        weight_down = batch.lower_laplacian_weight.clone() * tau_down

        for i in self.convergence_checker:
            # lower gradient steps
            for _ in range(k):
                f, delta_f = self._lower_gradient_step(
                    f, delta_f, index_down, weight_down, momentum_down, virtual, f_true
                )

            # transform to h
            h = self._transform_to_h(f, loss_coefficient)

            # upper_gradient steps
            h, delta_h = self._upper_gradient_step(
                h,
                delta_h,
                index_up,
                weight_up,
                momentum_up,
                known_nodes,
                reservoir_connector,
                h_known,
                batch,
            )

            f = self._transform_to_f(h, f, loss_coefficient, reservoir_connector)

            self.convergence_checker.update(
                batch=batch,
                f=f,
                h=h,
                delta_f=delta_f,
                delta_h=delta_h,
                virtual_nodes=virtual_nodes,
                i=i,
            )

        return f, h

    def _lower_gradient_step(
        self, f, delta_f, index, weight, momentum, virtual, f_known
    ):
        y_mask_old = f
        f = f - self.fp(f, index, weight) + momentum * delta_f
        # Apply virtual constraints
        f = torch.where(virtual, f_known, f)

        delta_f = f - y_mask_old
        return f, delta_f

    def _upper_gradient_step(
        self,
        h,
        delta_h,
        index,
        weight,
        momentum,
        known_nodes,
        reservoir_connector,
        h_known,
        batch,
    ):
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
