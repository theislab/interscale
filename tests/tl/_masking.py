import torch
import pytest
from torch_geometric.data import Data, Batch
from graph_transformer_long_range_niches.tl import apply_mask, MASK_VALUE

def test_apply_mask():
    # Create a simple graph with 4 nodes and features
    x = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    data = Data(x=x, edge_index=edge_index)
    
    # Convert to Batch (single graph)
    batch_data = Batch.from_data_list([data])
    
    # Create mask: mask the second node (index 1)
    batch_data.mask = torch.zeros(4, dtype=torch.bool)
    batch_data.mask[1] = True
    
    # Apply mask
    masked_data, mask_idx = apply_mask(batch_data)
    
    # Test assertions
    assert torch.equal(mask_idx, torch.tensor([1])), "Incorrect mask index returned"
    assert torch.equal(masked_data.x[1], torch.tensor([MASK_VALUE, MASK_VALUE])), "Node not properly masked"
    assert torch.equal(masked_data.x[0], torch.tensor([1., 2.])), "Unmasked node was modified"
    
def test_apply_mask_no_mask():
    # Test behavior when no mask is present
    x = torch.tensor([[1., 2.], [3., 4.]])
    data = Data(x=x)
    batch_data = Batch.from_data_list([data])
    batch_data.mask = None
    
    result = apply_mask(batch_data)
    assert result == batch_data, "Should return original data when no mask is present"