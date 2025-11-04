import pytest
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from yacs.config import CfgNode as CN

from InterScale.module.base import GlobalModuleClass
from InterScale.module.global_modules import TransformerNodeEncoderHook
from InterScale.tl import pad_batch, apply_mask
from InterScale.tl.masking import MASK_VALUE


@pytest.fixture
def sample_global_module_kwargs():
    """Create sample keyword arguments for global module testing."""
    return {
        'max_seq_len': 20,
        'n_heads': 2,
        'dropout_global': 0.1,
        'num_layers': 2,
        'dim_feedforward': 64,
        'long_range_attention': False
    }


@pytest.fixture
def sample_graph(num_nodes=10, num_features=5, num_classes=3, graph_level=True):
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
def sample_batch(sample_graph, num_graphs=2):
    """Create a batch of graphs for testing."""
    graphs = []
    for _ in range(num_graphs):
        graphs.append(sample_graph)
    batch = Batch.from_data_list(graphs)
    return batch

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
@pytest.mark.parametrize('pct_mask_nodes', [0.0, 0.2])
def test_common_step_local_to_global(sample_global_module_kwargs, sample_batch, pct_mask_nodes, prediction_level):
    """Test common_step_local_to_global."""
    n_input = 5
    n_output = 3
    n_embed = 4
    
    batch_size = len(np.unique(sample_batch.batch))

    module = TransformerNodeEncoderHook(
        n_input=n_input,
        n_output=n_output,
        n_embed=n_embed,
        decoder_type='linear',
        pct_mask_nodes=pct_mask_nodes,
        **sample_global_module_kwargs
    )
    
    batch_masked, mask_idx = module._common_step_masking(sample_batch)
    embedding = module.create_gex_embedding(batch_masked.x.cpu().numpy(), type="PCA")
    embedding = torch.tensor(embedding, dtype=torch.float32, device=batch_masked.x.device)
    assert embedding.shape == (batch_masked.x.shape[0], n_embed), f"Mismatch: embedding.shape: {embedding.shape}, batch_masked.x.shape: {batch_masked.x.shape}"
    
    padded_emb, src_padding_mask, pad_index_nodes, attention_mask = module.common_step_local_to_global(batch_masked, embedding)
    print('padded_emb', padded_emb.shape)
    print('src_padding_mask', src_padding_mask.shape)
    print('pad_index_nodes', len(pad_index_nodes), pad_index_nodes)
    if attention_mask is not None:
        print('attention_mask', attention_mask.shape, attention_mask)
    else:
        print('attention_mask is None')
    
    global_embedding, src_padding_mask = module.forward(padded_emb, src_padding_mask, attention_mask)
    y_pred = module.predict(global_embedding, src_padding_mask, prediction_level)

    print('y_pred', y_pred.shape, y_pred)
    print('embedding', embedding.shape, embedding)
    print('global_embedding', global_embedding.shape, global_embedding)
    #y_true, adjusted_mask_idx = module._process_batch_for_metrics(sample_batch, 'classification', prediction_level, pad_index_nodes, mask_idx)
    # print('y_true', y_true.shape, y_true)
    # print('adjusted_mask_idx', adjusted_mask_idx.shape, adjusted_mask_idx)
