"""Utility functions for testing InterScale models.

This module provides helper functions for creating test data and configurations
that can be reused across multiple test files.
"""
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from torch_geometric.data import Data

from InterScale.config.local_component_config import get_local_component_cfg
from InterScale.config.global_component_config import get_global_component_cfg
from InterScale.config import get_cfg_defaults
from yacs.config import CfgNode as CN


def create_minimal_adata(n_obs=50, n_vars=20, n_samples=2):
    """Create a minimal AnnData object for testing.
    
    Parameters
    ----------
    n_obs : int, optional
        Number of observations (cells/spots), by default 50
    n_vars : int, optional
        Number of variables (genes), by default 20
    n_samples : int, optional
        Number of samples to create, by default 2
        
    Returns
    -------
    AnnData
        AnnData object with required fields for model testing:
        - X: expression data
        - obsm['spatial']: spatial coordinates
        - obs['sample_key']: sample identifiers
        - obs['prediction_obs']: prediction labels (for classification)
        - obs['split']: train/val/test split labels
    """
    # Create random expression data
    X = np.random.randn(n_obs, n_vars)
    
    # Create spatial coordinates
    spatial = np.random.rand(n_obs, 2) * 100
    
    # Create sample keys (split data into samples)
    sample_keys = np.random.choice([f'sample_{i}' for i in range(n_samples)], n_obs)
    
    # Create prediction labels (for classification)
    prediction_obs = np.random.choice(['class_0', 'class_1'], n_obs)
    
    # Create group labels
    group_labels = np.random.choice(['group_0', 'group_1'], n_obs)
    # Create split labels
    split_labels = np.random.choice(['train', 'val', 'test'], n_obs, p=[0.7, 0.2, 0.1])
    
    adata = AnnData(X)
    adata.obsm['spatial'] = spatial
    adata.obs['sample_key'] = pd.Categorical(sample_keys)
    adata.obs['prediction_obs'] = pd.Categorical(prediction_obs)
    adata.obs['split'] = pd.Categorical(split_labels)
    adata.obs['group_label'] = pd.Categorical(group_labels)
    
    return adata


def sample_config(local_component_name=None, global_component_name=None, dual_decoder=False):
    """Create a test configuration for InterScale models.
    
    Parameters
    ----------
    local_component_name : str, optional
        Name of local component (e.g., 'GCN'), by default None
    global_component_name : str, optional
        Name of global component (e.g., 'self-attn-transformer'), by default None
    dual_decoder : bool, optional
        Whether to use dual decoder for CombinedModel, by default False
        
    Returns
    -------
    CN
        Frozen CfgNode with test configuration
    """
    cfg = get_cfg_defaults()
    
    # Load component configs if specified
    if local_component_name is not None:
        cfg = get_local_component_cfg(cfg, local_component_name)
    if global_component_name is not None:
        cfg = get_global_component_cfg(cfg, global_component_name)
    
    # Create custom config dict (without 'cfg' wrapper)
    custom_cfg = {
        'dataset': {
            'num_classes': 2,
            'num_features': 9,
            'prediction_task': 'classification',
            'prediction_level': 'node',
            'batch_size': 2,
            'group_label': 'group_label',
            'sample_key': ['sample_key'],
            'prediction_obs': 'prediction_obs',
            'layer_key': None,
            'split_key': 'split',
            'pct_mask_nodes': 0.2
        },
        'optim': {
            'lr': 0.001
        },
        'model': {
            'n_embed': 16,
            'decoder': {
                'type': 'linear',
                'dual_decoder': dual_decoder
            }
        }
    }
    
    # Merge custom config
    cfg.merge_from_other_cfg(CN(custom_cfg))
    
    # Set component names if provided
    if local_component_name is not None:
        cfg.model.local_component.name = local_component_name
    if global_component_name is not None:
        cfg.model.global_component.name = global_component_name
    
    cfg.freeze()
    return cfg


def sample_graph_data(sample_transformer_config):
    """Create sample graph data for testing."""
    num_nodes = sample_transformer_config['n_input']
    num_features = 40
    
    # Create random node features
    x = torch.randn(num_nodes, num_features)
    
    # Create sample edges (simple chain graph)
    edge_index = torch.tensor([[i, i+1] for i in range(num_nodes-1)], dtype=torch.long).t()
    
    # Create sample data object
    data = Data(x=x, edge_index=edge_index)
    data.obs_names = [f"node_{i}" for i in range(num_nodes)]
    data.batch = torch.zeros(num_nodes, dtype=torch.long)
    data.mask = torch.ones(num_nodes, dtype=torch.bool)
    
    return data

def create_test_pyg_data():
    """Creates a list of four PyG objects with different sizes and class distribution."""
    # Create data objects of different sizes
    sizes = [10, 20, 40, 100]
    data_list = []
    
    # Define class distribution
    class_distribution = {
        0: 0.50,  # 50% class 1
        1: 0.25,  # 25% class 2
        2: 0.20,  # 20% class 3
        3: 0.05   # 5% class 4
    }
    
    for size in sizes:
        # Create random features
        x = torch.randn(size, 10)  # 10 features per node
        
        # Create random edges (fully connected for simplicity)
        edge_index = []
        for i in range(size):
            for j in range(size):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index).t()
        
        # Create labels according to distribution
        num_nodes_per_class = {
            cls: int(size * prob) for cls, prob in class_distribution.items()
        }
        # Adjust for rounding errors
        remaining = size - sum(num_nodes_per_class.values())
        num_nodes_per_class[0] += remaining
        
        y = []
        for cls, num in num_nodes_per_class.items():
            y.extend([cls] * num)
        y = torch.tensor(y)
        
        # Create PyG data object
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    
    return data_list
