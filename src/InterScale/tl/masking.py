import torch
from torch_geometric.data import Batch

MASK_VALUE = 0  

def apply_mask(batched_data: Batch):
    """Mask nodes from PyG object in .mask attribute.

    Args:
        batched_data (Batch): _description_

    Returns:
        batched_data_w_mask (Batch): 
            Batch only containing nodes that were not masked
        mask_idx (torch.Tensor):
            Indices of masked nodes
            
    Example: 
        Data object:
        x = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        data = Data(x=x, edge_index=edge_index, mask=mask)
        ----
        mask_idx = torch.tensor([1, 3])
        masked_values = torch.tensor([[0., 0.], [3., 4.], [0., 0.], [7., 8.]])
    """
    if batched_data.mask is None:
        print("No mask found in batched_data")
        return batched_data
    mask = batched_data.mask
    mask_idx = torch.where(mask == 1)[0] # TODO into 2D array [B, N_batched_nodes]
    masked_values = batched_data.x.clone()
    masked_values[mask] = MASK_VALUE
    batched_data_w_mask = batched_data.clone()
    batched_data_w_mask.x = masked_values
    return batched_data_w_mask, mask_idx