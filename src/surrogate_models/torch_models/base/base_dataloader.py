from typing import Mapping, Sequence

import torch.utils.data.sampler
import torch_geometric
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData

from src.surrogate_models.torch_models.data.simplex import SimplexData


class Collater(torch_geometric.loader.dataloader.Collater):

    def __call__(self, batch):
        elem = batch[0]
        if isinstance(elem, SimplexData):
            return elem.from_data_list(batch, self.follow_batch,
                                       self.exclude_keys)
        elif isinstance(elem, BaseData):
            return Batch.from_data_list(batch, self.follow_batch,
                                        self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f'DataLoader found invalid type: {type(elem)}')

#
