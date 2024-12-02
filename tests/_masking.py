import torch
import pytest
import numpy as np
from torch_geometric.data import Data, Batch
from graph_transformer_long_range_niches.tl import mask_nodes

from yacs.config import CfgNode as CN
import pdb

from graph_transformer_long_range_niches.config import get_cfg_defaults

@pytest.fixture
def sample_config():
    cfg = get_cfg_defaults()
    custom_cfg = {
        'dataset': {
            'num_classes': 2,
            'num_features': 10,
            'prediction_task': 'graph_classification',
            'batch_size': 2
        }
    }
    cfg.merge_from_other_cfg(CN(custom_cfg))
    return cfg

def create_sample_graph(cfg, num_nodes=5, is_graph_level=True):
    # Create random node features
    num_features = cfg.dataset.num_features
    x = torch.randn(num_nodes, num_features)
    
    # Create sample edges (simple chain graph)
    edge_index = torch.tensor([[i, i+1] for i in range(num_nodes-1)], dtype=torch.long).t()
    
    if is_graph_level:
        # For graph-level tasks, use same label for all nodes
        y = torch.tensor([1] * num_nodes)  # Binary classification example
    else:
        # For node-level tasks, different labels per node
        y = torch.randint(0, 2, (num_nodes,))  # Binary classification example
    
    return Data(x=x, edge_index=edge_index, y=y)

def test_mask_nodes(sample_config):
    # Create sample data
    cfg = sample_config
    
    # Create sample batch
    batch = []
    for _ in range(cfg.dataset.batch_size):
        batch.append(create_sample_graph(cfg, num_nodes=np.random.randint(5, 10), is_graph_level=True))
    batch = Batch.from_data_list(batch)
    
    nr_nodes_to_mask = np.random.randint(1, 5)
    masked_batch, mask = mask_nodes(batch, nr_nodes_to_mask)
    
    pdb.set_trace()
    
    # Verify the output
    assert torch.sum(mask) == nr_nodes_to_mask  # Check if correct number of nodes are masked
    assert masked_batch.x.shape[0] == nr_nodes_to_mask  # Check if masked features have correct shape
    assert masked_batch.x.shape[1] == batch.x.shape[1]  # Check if feature dimension is preserved
    
if __name__ == "__main__":
    # This will run when you execute the file directly
    cfg = sample_config()
    test_mask_nodes(cfg)