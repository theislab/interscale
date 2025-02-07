from pathlib import Path
import pytest

import pytest
import torch
from torch_geometric.data import Data, Batch

from yacs.config import CfgNode as CN

from graph_transformer_long_range_niches.config import get_cfg_defaults

HERE: Path = Path(__file__).parent

@pytest.fixture
def sample_config():
    cfg = get_cfg_defaults()
    custom_cfg = {
        'dataset': {
            'num_classes': 2,
            'num_features': 9,
            'prediction_task': 'graph_classification',
            'batch_size': 2
        },
        'optim': {
            'lr': 0.001
        },
        'transformer': {
            'max_seq_len': 100
        }
    }
    cfg.merge_from_other_cfg(CN(custom_cfg))
    return cfg

@pytest.fixture
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

