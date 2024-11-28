import pytest
import torch
from torch_geometric.data import Data, Batch
import numpy as np
from yacs.config import CfgNode as CN

from graph_transformer_long_range_niches.config import load_config, get_cfg_defaults
import graph_transformer_long_range_niches as InterScale
#from graph_transformer_long_range_niches.model import LitGNNTransformer

def test_package_has_version():
    assert InterScale.__version__ is not None

@pytest.fixture
def sample_config():
    cfg = get_cfg_defaults()
    custom_cfg = {
        'dataset': {
            'num_classes': 2,
            'num_features': 10,
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

@pytest.mark.skip(reason="This decorator should be removed when test passes.")
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

#@pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_gnn_transformer_graph_classification(sample_config):
    # Set up model for graph-level prediction
    cfg = sample_config
    cfg.dataset.prediction_task = 'graph_classification'
    
    model = InterScale.model.LitGNNTransformer(cfg)

    # Create sample batch
    graphs = []
    for i in range(cfg.dataset.batch_size):
        graphs.append(create_sample_graph(cfg, num_nodes=np.random.randint(5, 10), is_graph_level=True))
    
    batch = Batch.from_data_list(graphs)
    
    # Test forward pass
    z, out, index_nodes = model(batch)
    
    # Check output shapes
    assert out.shape[0] == cfg.dataset.batch_size  # Batch size
    assert out.shape[1] == cfg.dataset.num_classes  # Number of classes
    assert z is not None
    assert index_nodes is not None

# def test_gnn_transformer_node_classification(sample_config):
#     # Set up model for node-level prediction
#     sample_config.prediction_task = 'node_classification'
#     model = LitGNNTransformer(sample_config)
    
#     # Create sample batch
#     graph1 = create_sample_graph(num_nodes=5, num_features=10, is_graph_level=False)
#     graph2 = create_sample_graph(num_nodes=4, num_features=10, is_graph_level=False)
#     batch = Batch.from_data_list([graph1, graph2])
    
#     # Test forward pass
#     z, out, index_nodes = model(batch)
    
#     # Check that output has predictions for each node
#     assert out.shape[1] == sample_config.num_classes  # Number of classes
#     assert z is not None
#     assert index_nodes is not None