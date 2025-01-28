import pytest
import torch
from torch_geometric.data import Data
from graph_transformer_long_range_niches.tl.geome_dataloader import GraphAnnDataModule

def create_toy_graph(num_nodes=10, num_features=5):
    """Create a simple toy graph for testing."""
    # Create random node features
    x = torch.randn(num_nodes, num_features)
    
    # Create a simple edge index (connecting each node to next node in sequence)
    edge_index = torch.tensor([[i, (i + 1) % num_nodes] for i in range(num_nodes)], 
                            dtype=torch.long).t()
    
    # Create random node labels
    y = torch.randn(num_nodes, num_features)
    
    return Data(x=x, edge_index=edge_index, y=y)

def test_graph_ann_data_module():
    # Create toy datasets
    train_graphs = [create_toy_graph(10, 5) for _ in range(3)]
    val_graphs = [create_toy_graph(10, 5) for _ in range(2)]
    test_graphs = [create_toy_graph(10, 5) for _ in range(2)]
    
    # Initialize the data module
    data_module = GraphAnnDataModule(
        datas=[train_graphs, val_graphs, test_graphs],
        batch_size=2,
        num_workers=0,  # Use 0 for testing
        pct_mask_nodes=0.5,
        learning_type="node"
    )
    
    # Test setup
    data_module.setup(stage="fit")
    assert data_module.setup_called == True
    
    # Get dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Basic checks
    for batch in train_loader:
        assert isinstance(batch, Data)
        assert hasattr(batch, 'mask')
        assert batch.mask.dtype == torch.bool
        assert torch.sum(batch.mask) > 0  # At least one node should be masked
        break
    
    # Test with invalid stage
    with pytest.raises(ValueError):
        data_module.setup(stage="invalid")
    
    # Test with invalid learning type
    with pytest.raises(ValueError):
        GraphAnnDataModule(
            datas=[train_graphs, val_graphs],
            learning_type="invalid"
        )

if __name__ == "__main__":
    test_graph_ann_data_module()