from typing import Optional

import numpy as np
import torch

from sklearn.metrics import mean_absolute_error
from torch.nn.functional import l1_loss
# from torch.ignite.metrics import Metric


def accuracy(output, target, *args, **kwargs):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3, *args, **kwargs):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

#@profile
def mae(output, target, mask: Optional[np.ndarray] = None, *args, **kwargs):
    with torch.no_grad():
        if mask is not None:
            if mask.sum() > 0:
                mask = mask.reshape(output.shape)
                return masked_mae(output, target, mask, *args, **kwargs)
            else:
                return 0
        # check if the output is a tensor
        if isinstance(output, torch.Tensor):
            loss = l1_loss(output, target)
        else:
            loss = mean_absolute_error(output, target)
    return loss



def mape(output, target, *args, **kwargs):
    with torch.no_grad():
        loss = np.average(np.abs(output - target) / np.abs(target), axis=0)
    return loss



def relative_ae(output, target, base=None, *args, **kwargs):
    if base is None:
        base = target
    with torch.no_grad():
        loss = np.average(np.abs(output - target) / target, axis=0)
    return loss


def masked_mae(output, target, mask, *args, **kwargs):
    with torch.no_grad():
        loss = l1_loss(output * mask, target * mask) * mask.shape[0] / mask.sum()
    return loss


def rmse(output, target, *args, **kwargs):
    with torch.no_grad():
        loss = mse_loss(output, target)
        loss = torch.sqrt(loss)
    return loss


def std(output, target, *args, **kwargs):
    with torch.no_grad():
        loss = torch.std(output - target)
    return loss

"""
From https://en.wikipedia.org/wiki/Coefficient_of_determination
"""
def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def r2_score_np(output, target):
    target_mean = np.mean(target)
    ss_tot = np.sum((target - target_mean) ** 2)
    ss_res = np.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


class MaskedMAE:
    def __init__(self, key):
        self.key = key

    def __call__(self, *args, **kwargs):
        return masked_mae(*args, **kwargs)


def mae_per_graph(output, target, batch_indices, ptr):
    with torch.no_grad():
        res = np.bincount(batch_indices, weights=np.abs(output - target))
        graph_sizes = np.diff(ptr, axis=0)
    return res / graph_sizes


def r2_per_graph(output, target, batch_indices, ptr):
    with torch.no_grad():
        res = np.bincount(batch_indices, weights=np.abs(output - target))
        graph_sizes = np.diff(ptr, axis=0)
    return res / graph_sizes


def error_distribution(output: torch.Tensor,
                       target: torch.Tensor,
                       mask: Optional[torch.Tensor] = None):
    with torch.no_grad():
        if mask is not None:
            if mask.sum() > 0:
                mask = np.expand_dims(mask, (-1))
                output = output[mask]
                target = target[mask]
        difference = output - target
        return difference


def output_distribution(output: torch.Tensor, target: torch.Tensor,
                        mask: Optional[torch.Tensor] = None):
    with torch.no_grad():
        if mask is not None:
            if mask.sum() > 0:
                mask = np.expand_dims(mask, (-1))
                output = output[mask]
        return output
