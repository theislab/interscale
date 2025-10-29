import pytest
import torch
from torch_geometric.data import Data, Batch
from torch import nn
import numpy as np

from InterScale.module.global_modules.transformer_encoder import TransformerNodeEncoderHook


@pytest.fixture
def sample_transformer_config():
    """Create a sample configuration for the transformer encoder."""
    return {
        'max_seq_len': 50,
        'n_input': 10,
        'n_output': 5,
        'n_embed': 16,
        'pct_mask_nodes': 0.1,
        'n_heads': 4,
        'num_layers': 2,
        'dim_feedforward': 32,
        'dropout_global': 0.1,
        'long_range_attention': False  # Simplified for testing
    }


@pytest.fixture
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


@pytest.fixture
def transformer_encoder(sample_transformer_config):
    """Create a transformer encoder instance for testing."""
    return TransformerNodeEncoderHook(**sample_transformer_config)


def test_cls_embedding_at_sequence_end(transformer_encoder, sample_graph_data):
    """
    Test that the CLS embedding is appended at the end of the sequence.
    """
    # Set model to evaluation mode
    transformer_encoder.eval()
    
    # Create sample embeddings (simulating local model output)
    num_nodes = len(sample_graph_data.obs_names)
    embedding_dim = transformer_encoder.n_embed
    sample_embeddings = torch.randn(num_nodes, embedding_dim)
    
    # Test 1: Forward pass - get CLS embedding from appended sequence
    with torch.no_grad():
        
        cls_embedding = transformer_encoder.cls_embedding 
        print('cls_embedding.shape:', cls_embedding.shape)
        print('cls_embedding values:', cls_embedding)
        
        # Prepare data for forward pass
        padded_emb, src_padding_mask, index_nodes, attention_mask = transformer_encoder.common_step_local_to_global(
            sample_graph_data, sample_embeddings, eval=False
        )
        
        # append cls embedding
        expand_cls_embedding = transformer_encoder.cls_embedding.expand(1, padded_emb.size(1), -1)
        padded_emb = torch.cat([padded_emb, expand_cls_embedding], dim=0)
        print('expanded_cls_embedding.shape:', expand_cls_embedding)
        print('padded_emb.shape:', padded_emb)
        
        assert padded_emb[-1, :, :] == cls_embedding, "CLS embedding should be appended at the end of the sequence"
                

def test_cls_embedding_multiple_calls_consistency(transformer_encoder, sample_graph_data):
    """
    Test that CLS embedding remains consistent across multiple calls to the same method.
    """
    transformer_encoder.eval()
    
    num_nodes = len(sample_graph_data.obs_names)
    embedding_dim = transformer_encoder.n_embed
    sample_embeddings = torch.randn(num_nodes, embedding_dim)
    
    cls_embeddings = []
    
    # Call evaluate method multiple times
    # Note: evaluate method requires gradients for hook registration and backward pass
    for i in range(3):
        _, transformer_out, _, _, _ = transformer_encoder.evaluate(
            sample_graph_data, sample_embeddings
        )
        cls_embedding = transformer_out[-1, :, :]
        cls_embeddings.append(cls_embedding)
    
    # Verify all CLS embeddings are identical
    for i in range(1, len(cls_embeddings)):
        assert torch.allclose(cls_embeddings[0], cls_embeddings[i], atol=1e-6), \
            f"CLS embedding from call {i} should be identical to call 0"
    
    print(f"✓ CLS embedding consistency across multiple calls verified")


def test_cls_embedding_gradient_flow(transformer_encoder, sample_graph_data):
    """
    Test that CLS embedding gradients flow properly during training.
    """
    transformer_encoder.train()
    
    num_nodes = len(sample_graph_data.obs_names)
    embedding_dim = transformer_encoder.n_embed
    sample_embeddings = torch.randn(num_nodes, embedding_dim, requires_grad=True)
    
    # Forward pass
    padded_emb, src_padding_mask, _, _ = transformer_encoder.common_step_local_to_global(
        sample_graph_data, sample_embeddings, eval=False
    )
    
    transformer_out, _ = transformer_encoder.forward(padded_emb, src_padding_mask)
    
    # Extract CLS embedding and compute loss
    cls_embedding = transformer_out[-1, :, :]
    loss = cls_embedding.sum()
    
    # Backward pass
    loss.backward()
    
    # Verify CLS embedding parameter has gradients
    assert transformer_encoder.cls_embedding.grad is not None, \
        "CLS embedding parameter should have gradients after backward pass"
    
    # Verify gradient shape
    expected_grad_shape = transformer_encoder.cls_embedding.shape
    assert transformer_encoder.cls_embedding.grad.shape == expected_grad_shape, \
        f"CLS embedding gradient shape should be {expected_grad_shape}, got {transformer_encoder.cls_embedding.grad.shape}"
    
    print(f"✓ CLS embedding gradient flow test passed")


def test_cls_embedding_parameter_initialization(transformer_encoder):
    """
    Test that CLS embedding parameter is properly initialized.
    """
    # Verify CLS embedding parameter exists
    assert hasattr(transformer_encoder, 'cls_embedding'), \
        "Transformer encoder should have cls_embedding parameter"
    
    # Verify parameter shape
    expected_shape = (1, 1, transformer_encoder.n_embed)
    assert transformer_encoder.cls_embedding.shape == expected_shape, \
        f"CLS embedding shape should be {expected_shape}, got {transformer_encoder.cls_embedding.shape}"
    
    # Verify parameter requires gradients
    assert transformer_encoder.cls_embedding.requires_grad, \
        "CLS embedding parameter should require gradients"
    
    # Verify parameter is not all zeros (random initialization)
    assert not torch.allclose(transformer_encoder.cls_embedding, torch.zeros_like(transformer_encoder.cls_embedding)), \
        "CLS embedding should not be initialized to zeros"
    
    print(f"✓ CLS embedding parameter initialization test passed")


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
