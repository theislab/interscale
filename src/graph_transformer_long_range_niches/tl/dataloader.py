from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Any, Optional

from torch_geometric.data import Dataset
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.loader import DataLoader

import torch
import random
import numpy as np

def smallest_data_batch_length(data_batch):
    # Use `batch` to identify which nodes belong to each graph
    batch_size = data_batch.batch.max().item() + 1  # Number of graphs
    lengths = [(data_batch.batch == i).sum().item() for i in range(batch_size)]
    return min(lengths)

# class MaskedNodeLightningDataset(LightningDataset):
#     def __init__(self, train_dataset: Dataset,
#                         val_dataset: Optional[Dataset] = None,
#                         test_dataset: Optional[Dataset] = None,
#                         pred_dataset: Optional[Dataset] = None,
#                         pct_mask_nodes: float = 0.7,
#                         **kwargs: Any):
#         super().__init__(train_dataset, val_dataset, test_dataset, pred_dataset, **kwargs)
#         self.pct_mask_nodes = pct_mask_nodes
    
#     def dataloader(self, dataset: Dataset, **kwargs: Any) -> DataLoader:
#         return MaskEveryNodeLoader(dataset, pct_mask_nodes=self.pct_mask_nodes, **kwargs)

class MaskedNodeLightningDataset(LightningDataset):
    def __init__(self, pct_mask_nodes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pct_mask_nodes = pct_mask_nodes
    
    def dataloader(self, dataset: Dataset, **kwargs: Any) -> DataLoader:
        return MaskEveryNodeLoader(dataset, pct_mask_nodes=self.pct_mask_nodes, **kwargs)

class MaskEveryNodeLoader(DataLoader):
    def __init__(self, data, pct_mask_nodes=0.7, shuffle=False, **kwargs):
        """
        A DataLoader that dynamically masks one node at a time.
        
        Parameters:
        - data (torch_geometric.data.Data): The graph data to mask.
        - pct_mask_nodes (float): The percentage of nodes to mask.
        - shuffle (bool): Whether to shuffle the node order.
        """
        super().__init__(data, **kwargs)
        self.data = data
        self.num_nodes = data.num_nodes
        self.smallest_graph_num_nodes = self.smallest_data_batch_length(self.data)
        self.indices = torch.arange(self.num_nodes)
        self.pct_mask_nodes = pct_mask_nodes
        if shuffle:
            self.indices = self.indices[torch.randperm(self.num_nodes)]

    def __len__(self):
        return self.num_nodes
    
    def smallest_data_batch_length(self, data_batch):
        batch_size = data_batch.batch.max().item() + 1  # Number of graphs
        lengths = [(data_batch.batch == i).sum().item() for i in range(batch_size)]
        return min(lengths)

    def __iter__(self):
        # To Do: Change equation: fails for very different number of cells in graph
        num_nodes = int(self.smallest_graph_num_nodes * self.pct_mask_nodes)
        index_list = []
        for batch_idx in np.unique(self.data.batch):
            mask = self.data.batch.eq(batch_idx)
            indices = np.where(mask)[0]
            start = indices[0] if len(indices) > 0 else None
            end = indices[-1] if len(indices) > 0 else None
            assert (end - start) > num_nodes # don't sample more nodes than available in a batch
            index_list.append(random.sample(range(start, end), num_nodes))
        for idx in range(num_nodes):
            masked_data = self.data.clone()
            masked_data.mask = torch.zeros(self.num_nodes, dtype=torch.bool)
            mask_index = [sublist[idx] for sublist in index_list]
            masked_data.mask[mask_index] = True
            yield masked_data