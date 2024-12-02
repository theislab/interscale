import torch
from torch_geometric.data import Batch

MASK_VALUE = 1  

def mask_nodes(batched_data: Batch, nr_nodes: int):
    """Mask a specific number of nodes in the batch.

    Args:
        batched_data (Batch): _description_
        nr_nodes (int): _description_

    Returns:
        batched_data (Batch): 
            Batch only containing nodes that were not masked
        mask (torch.Tensor): 
            Mask indicating which nodes were masked (MASK_VALUE)
    """
    mask = torch.zeros(batched_data.num_nodes)
    node_idx = torch.randperm(batched_data.num_nodes)[:nr_nodes]
    mask[node_idx] = MASK_VALUE 
    batched_data.x = batched_data.x[mask==MASK_VALUE]
    return batched_data, mask