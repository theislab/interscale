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
    
    
def test_apply_mask_with_mask():
    """Test apply_mask when a mask is provided."""
    # Create a simple batch with 3 nodes
    x = torch.randn(3, 4)  # 3 nodes with 4 features each
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    batch = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
    
    # Create a mask that masks the second node
    batch.mask = torch.tensor([0, 1, 0], dtype=torch.bool)
    
    # Apply mask
    masked_batch, mask_idx = apply_mask(batch)
    
    # Check that the masked node's values are set to MASK_VALUE
    assert torch.allclose(masked_batch.x[1], torch.zeros(4) * MASK_VALUE)
    assert not torch.allclose(masked_batch.x[0], torch.zeros(4) * MASK_VALUE)
    assert not torch.allclose(masked_batch.x[2], torch.zeros(4) * MASK_VALUE)
    
    # Check that mask_idx contains the correct index
    assert mask_idx.tolist() == [1]

def test_apply_mask_multiple_masks():
    """Test apply_mask with multiple masked nodes."""
    # Create a simple batch with 4 nodes
    x = torch.randn(4, 4)  # 4 nodes with 4 features each
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    batch = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
    
    # Create a mask that masks the first and third nodes
    batch.mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    
    # Apply mask
    masked_batch, mask_idx = apply_mask(batch)
    
    # Check that the masked nodes' values are set to MASK_VALUE
    assert torch.allclose(masked_batch.x[0], torch.zeros(4) * MASK_VALUE)
    assert torch.allclose(masked_batch.x[2], torch.zeros(4) * MASK_VALUE)
    assert not torch.allclose(masked_batch.x[1], torch.zeros(4) * MASK_VALUE)
    assert not torch.allclose(masked_batch.x[3], torch.zeros(4) * MASK_VALUE)
    
    # Check that mask_idx contains the correct indices
    assert sorted(mask_idx.tolist()) == [0, 2]

def test_apply_mask_batch_multiple_graphs():
    """Test apply_mask with a batch containing multiple graphs."""
    # Create two graphs with 3 nodes each
    x1 = torch.randn(3, 4)
    x2 = torch.randn(3, 4)
    edge_index1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_index2 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    
    batch = Batch.from_data_list([
        Data(x=x1, edge_index=edge_index1),
        Data(x=x2, edge_index=edge_index2)
    ])
    
    # Create a mask that masks one node from each graph
    batch.mask = torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.bool)
    
    # Apply mask
    masked_batch, mask_idx = apply_mask(batch)
    
    # Check that the masked nodes' values are set to MASK_VALUE
    assert torch.allclose(masked_batch.x[1], torch.zeros(4) * MASK_VALUE)
    assert torch.allclose(masked_batch.x[4], torch.zeros(4) * MASK_VALUE)
    
    # Check that mask_idx contains the correct indices
    assert sorted(mask_idx.tolist()) == [1, 4] 