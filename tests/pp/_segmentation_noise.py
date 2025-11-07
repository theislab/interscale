#!/usr/bin/env python3
"""
Test script for segmentation noise functionality
"""
import pytest
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix
from InterScale.pp.segmentation_noise import apply_segmentation_noise
from InterScale.config import get_cfg_defaults
from InterScale.tl import prepare_geome_dataset


def create_test_data(n_cells=100, n_genes=50):
    """Create a simple test AnnData object with spatial adjacency"""
    
    # Create random gene expression data
    X = np.random.poisson(5, size=(n_cells, n_genes)).astype(float)
    
    # Create spatial coordinates (cells in a grid)
    n_rows = int(np.sqrt(n_cells))
    n_cols = n_cells // n_rows
    
    coords = []
    for i in range(n_cells):
        row = i // n_cols
        col = i % n_cols
        coords.append([row * 10, col * 10])  # 10 unit spacing
    
    coords = np.array(coords)
    
    # Create adjacency matrix (4-connected grid)
    adj_matrix = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        row_i, col_i = i // n_cols, i % n_cols
        
        # Check neighbors
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row_i + di, col_i + dj
            if 0 <= new_row < n_rows and 0 <= new_col < n_cols:
                j = new_row * n_cols + new_col
                adj_matrix[i, j] = 1
    
    # Create AnnData object
    adata = sc.AnnData(X=X)
    
    return adata


@pytest.mark.parametrize("node_fraction", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("overflow_fraction", [0.1, 0.3, 0.5])
def test_segmentation_noise(node_fraction, overflow_fraction):
    np.random.seed(0)
    
    """Test the segmentation noise function"""
    
    adata = create_test_data(n_cells=50, n_genes=20)
    adata_mean = adata.X.mean()
    
    adata.obsm['spatial'] = np.random.rand(50, 2)
    # Create random sparse adjacency matrix with ~10% connections
    random_adj = np.random.choice([0, 1], size=(50, 50), p=[0.6, 0.4])
    random_adj = np.maximum(random_adj, random_adj.T)
    adata.obsp['adjacency_matrix_connectivities'] = csr_matrix(random_adj)
    
    
    adata_noisy = apply_segmentation_noise(
        adata, 
        node_fraction=node_fraction, 
        overflow_fraction=overflow_fraction
    )
    
    # original adata object is not modified
    assert adata.X.mean() == adata_mean
    # different distribution when segmentation applied    
    assert adata_noisy.X.mean() != adata.X.mean()
    
    
@pytest.mark.parametrize("node_fraction", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("overflow_fraction", [0.1, 0.3, 0.5])
def test_segmentation_noise_geome_loading(node_fraction, overflow_fraction):
    adata = create_test_data(n_cells=50, n_genes=20)
    cfg = get_cfg_defaults()
    cfg.dataset.segmentation_robustness = [node_fraction, overflow_fraction]
    pyg_data_list, _ = prepare_geome_dataset(adata, cfg)

