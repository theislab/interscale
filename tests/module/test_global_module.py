import pytest
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from yacs.config import CfgNode as CN

from InterScale.module.base import GlobalModuleClass
from InterScale.module.global_modules import TransformerNodeEncoderHook
from InterScale.tl import pad_batch, apply_mask
from InterScale.tl.masking import MASK_VALUE


def create_sample_global_module_kwargs(long_range_attention):
    """Create sample keyword arguments for global module testing."""
    return {
        'max_seq_len': 20,
        'n_heads': 2,
        'dropout_global': 0.1,
        'num_layers': 2,
        'dim_feedforward': 64,
        'long_range_attention': long_range_attention
    }


def create_sample_graph(num_nodes=10, num_features=5, num_classes=3, graph_level=True):
    """Create a sample PyG graph for testing."""
    x = torch.randn(num_nodes, num_features)
    
    # Create simple chain edges
    edge_index = torch.tensor([[i, i+1] for i in range(num_nodes-1)], dtype=torch.long).t()
    # Make it undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    if graph_level:
        # For graph-level: same label for all nodes (one-hot encoded for classification)
        y = torch.zeros(num_nodes, num_classes)
        class_idx = torch.randint(0, num_classes, (1,)).item()
        y[:, class_idx] = 1.0
    else:
        # For node-level: different labels per node
        y = torch.zeros(num_nodes, num_classes)
        for i in range(num_nodes):
            class_idx = torch.randint(0, num_classes, (1,)).item()
            y[i, class_idx] = 1.0
    
    data = Data(x=x, edge_index=edge_index, y=y)
    data.obs_names = torch.arange(num_nodes)
    data.mask = torch.randint(0, 2, (num_nodes,), dtype=torch.bool)
    
    return data


@pytest.fixture
def sample_batch(num_graphs=2, num_features=5, num_classes=3, graph_level=True):
    """Create a batch of graphs for testing."""
    graphs = []
    for _ in range(num_graphs):
        num_nodes = np.random.randint(5, 11)
        sample_graph = create_sample_graph(num_nodes=num_nodes, num_features=num_features, num_classes=num_classes, graph_level=graph_level)
        graphs.append(sample_graph)
    batch = Batch.from_data_list(graphs)
    return batch


def create_module(sample_batch, long_range_attention=False, pct_mask_nodes=0.0):
    """Create a batch of graphs for testing."""
    n_input = 5
    n_output = 3
    n_embed = 4
    
    batch_size = len(np.unique(sample_batch.batch))
    
    global_module_kwargs = create_sample_global_module_kwargs(long_range_attention=long_range_attention)

    module = TransformerNodeEncoderHook(
        n_input=n_input,
        n_output=n_output,
        n_embed=n_embed,
        decoder_type='linear',
        pct_mask_nodes=pct_mask_nodes,
        **global_module_kwargs
    )
    
    return module

@pytest.mark.parametrize('pct_mask_nodes', [0.0, 0.2])
def test_common_step_masking(sample_global_module_kwargs, sample_batch, pct_mask_nodes):
    """Test _common_step_masking."""
    n_input = 5
    n_output = 3
    n_embed = 4
    
    module = TransformerNodeEncoderHook(
        n_input=n_input,
        n_output=n_output,
        n_embed=n_embed,
        decoder_type='linear',
        pct_mask_nodes=pct_mask_nodes,
        **sample_global_module_kwargs
    )
    
    print('sample_batch.mask', sample_batch.mask)
    batch_masked, mask_idx = module._common_step_masking(sample_batch)
    print('batch_masked', batch_masked)
    print('mask_idx', mask_idx)
        
    if pct_mask_nodes > 0:
        assert mask_idx.shape == (sample_batch.mask.sum().item(),)
    else:
        assert mask_idx.shape == (sample_batch.x.shape[0],)
        
@pytest.mark.parametrize('prediction_level', ['node', 'graph'])
@pytest.mark.parametrize('pct_mask_nodes', [0.0, 0.6])
@pytest.mark.parametrize('long_range_attention', [True, False])
def test_common_step_local_to_global(sample_batch, pct_mask_nodes, prediction_level, long_range_attention):
    """Test common_step_local_to_global."""
    n_input = 5
    n_output = 3
    n_embed = 4
    
    batch_size = len(np.unique(sample_batch.batch))
    
    global_module_kwargs = create_sample_global_module_kwargs(long_range_attention=long_range_attention)

    module = TransformerNodeEncoderHook(
        n_input=n_input,
        n_output=n_output,
        n_embed=n_embed,
        decoder_type='linear',
        pct_mask_nodes=pct_mask_nodes,
        **global_module_kwargs
    )
    
    batch_masked, mask_idx = module._common_step_masking(sample_batch)
    embedding = module.create_gex_embedding(batch_masked.x.cpu().numpy(), type="PCA")
    embedding = torch.tensor(embedding, dtype=torch.float32, device=batch_masked.x.device)
    assert embedding.shape == (batch_masked.x.shape[0], n_embed), f"Mismatch: embedding.shape: {embedding.shape}, batch_masked.x.shape: {batch_masked.x.shape}"
    
    padded_emb, src_padding_mask, pad_index_nodes, attention_mask = module.common_step_local_to_global(batch_masked, embedding)
    print('padded_emb', padded_emb.shape, padded_emb)
    if attention_mask is not None:
        print('attention_mask', attention_mask.shape, attention_mask)
    else:
        print('attention_mask is None')
        
    global_embedding, src_padding_mask = module.forward(padded_emb, src_padding_mask, attention_mask)
    print('src_padding_mask', src_padding_mask.shape, src_padding_mask)
    
    y_pred = module.predict(global_embedding, src_padding_mask, prediction_level)
    
    print('y_pred', y_pred.shape, y_pred)
    print('embedding', embedding.shape, embedding)
    print('global_embedding', global_embedding.shape, global_embedding)
    
    assert not torch.any(torch.isnan(global_embedding)), "global_embedding contains NaN values"
    assert not torch.any(torch.isnan(y_pred)), "y_pred contains NaN values"


def test_evaluate(sample_batch, long_range_attention=False):
    """Test the .evaluate() method. No masking during evaluation.
    """
    module = create_module(sample_batch, long_range_attention=long_range_attention)
    
    embedding = module.create_gex_embedding(sample_batch.x.cpu().numpy(), type="PCA")
    embedding = torch.tensor(embedding, dtype=torch.float32, device=sample_batch.x.device)
    transformer_in, transformer_out, eval_src_padding_mask, eval_pad_index_nodes, I = module.evaluate(sample_batch, embedding)

    assert I.shape[0] == I.shape[1], f"Relevance matrix returned by evaluate() should be square, got {I.shape}"

    # Check type and nan safety
    assert isinstance(I, torch.Tensor)
    assert not torch.any(torch.isnan(I)), "Relevance matrix I contains NaN values"
    assert not torch.any(torch.isnan(transformer_out)), "transformer_out contains NaN values"

def test_process_batch_for_metrics_mask_idx_exploration():
    """Test function to explore the relationship between mask_idx and pad_index_nodes.
    
    This test explores whether mask_idx can be correctly sliced to match pad_index_nodes,
    which is an assumption in _process_batch_for_metrics.
    """
    from InterScale.module.base._base_global_module import GlobalModuleClass
    
    # Create a mock batch with known structure
    # Batch 0: 10 nodes (indices 0-9)
    # Batch 1: 8 nodes (indices 10-17)
    # Batch 2: 12 nodes (indices 18-29)
    num_nodes_per_batch = [10, 8, 12]
    total_nodes = sum(num_nodes_per_batch)
    
    # Create batch tensor
    batch_tensor = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(num_nodes_per_batch)])
    
    # Create mock batch object
    class MockBatch:
        def __init__(self):
            self.batch = batch_tensor
            self.x = torch.randn(total_nodes, 5)
            self.y = torch.randint(0, 3, (total_nodes, 3)).float()
            self.ptr = torch.tensor([0, 10, 18, 30], dtype=torch.long)
    
    batch = MockBatch()
    
    # Scenario 1: No nodes masked (pct_mask_nodes = 0.0 means no nodes are "masked" for prediction)
    # mask_idx should be empty: []
    mask_idx_all = torch.tensor([])
    
    # Scenario 2: Partial masking - some nodes masked in each batch
    # Batch 0: mask nodes [1, 3, 5, 7, 9] (global indices)
    # Batch 1: mask nodes [11, 13, 15] (global indices)
    # Batch 2: mask nodes [19, 21, 23, 25, 27, 29] (global indices)
    mask_idx_partial = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15, 19, 21, 23, 25, 27, 29])
    
    # Scenario 3: Different number of masked nodes per batch
    # Batch 0: mask 3 nodes [2, 4, 6]
    # Batch 1: mask 5 nodes [10, 12, 14, 16, 17]
    # Batch 2: mask 2 nodes [20, 24]
    mask_idx_uneven = torch.tensor([2, 4, 6, 10, 12, 14, 16, 17, 20, 24])
    
    embedding = batch.x
    
    for mask_idx, scenario_name in zip([mask_idx_all, mask_idx_partial, mask_idx_uneven], ["All nodes masked", "Partial masking", "Different number of masked nodes per batch"]):
        print(f"Scenario: {scenario_name}")
        max_seq_len = 10
        padded_emb, src_padding_mask, index_nodes, num_nodes, mask, max_num_nodes = pad_batch(
                embedding, 
                batch.batch, 
                max_seq_len, 
                get_mask=True, 
                keep_indices=mask_idx  # Add parameter to ensure masked nodes are kept (not during evaluation) 
            )
        
        y_true, adjusted_mask_idx = module._process_batch_for_metrics(batch, prediction_task, prediction_level, index_nodes, mask_idx)
        
    
    
