import pytest
import torch
from torch_geometric.data import Data, Batch
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
def sample_graph_data():
    """Create sample graph data for testing."""
    num_nodes = 8
    num_features = 10
    
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


def test_cls_embedding_consistency_forward_vs_evaluate(transformer_encoder, sample_graph_data):
    """
    Test that the CLS embedding appended during forward pass is the same 
    as the one retrieved during evaluation.
    
    This test verifies:
    1. CLS embedding is properly appended during forward pass
    2. CLS embedding is positioned correctly (at the end of sequence)
    3. The same CLS embedding is used in both forward and evaluate methods
    4. CLS embedding values are consistent across calls
    """
    # Set model to evaluation mode
    transformer_encoder.eval()
    
    # Create sample embeddings (simulating local model output)
    num_nodes = len(sample_graph_data.obs_names)
    embedding_dim = transformer_encoder.n_embed
    sample_embeddings = torch.randn(num_nodes, embedding_dim)
    
    # Test 1: Forward pass - get CLS embedding from appended sequence
    with torch.no_grad():
        # Prepare data for forward pass
        padded_emb, src_padding_mask, index_nodes, attention_mask = transformer_encoder.common_step_local_to_global(
            sample_graph_data, sample_embeddings, eval=False
        )
        
        # Forward pass
        transformer_out_forward, src_padding_mask_forward = transformer_encoder.forward(
            padded_emb, src_padding_mask, register_hook=False
        )
        
        print('transformer_out_forward.shape:', transformer_out_forward.shape)
        
        # Extract CLS embedding from forward pass output (last position)
        cls_embedding_forward = transformer_out_forward[-1, :, :]  # Shape: [batch_size, n_embed]
        
    # Test 2: Evaluate method - get CLS embedding from evaluation
    # Note: evaluate method requires gradients for hook registration and backward pass
    transformer_in_eval, transformer_out_eval, src_padding_mask_eval, index_nodes_eval, I = transformer_encoder.evaluate(
        sample_graph_data, sample_embeddings
    )
    
    # Extract CLS embedding from evaluate output (last position)
    cls_embedding_eval = transformer_out_eval[-1, :, :]  # Shape: [batch_size, n_embed]
    
    # Test 3: Verify CLS embeddings are identical
    assert torch.allclose(cls_embedding_forward, cls_embedding_eval, atol=1e-6), \
        "CLS embeddings from forward and evaluate methods should be identical"
    
    # Test 4: Verify CLS embedding is the same as the stored parameter
    stored_cls_embedding = transformer_encoder.cls_embedding.squeeze(0)  # Remove batch dimension
    assert torch.allclose(cls_embedding_forward, stored_cls_embedding, atol=1e-6), \
        "CLS embedding should match the stored parameter"
    
    # Test 5: Verify output shapes are correct
    batch_size = 1  # Single graph
    assert cls_embedding_forward.shape == (batch_size, embedding_dim), \
        f"CLS embedding shape should be ({batch_size}, {embedding_dim}), got {cls_embedding_forward.shape}"
    
    assert cls_embedding_eval.shape == (batch_size, embedding_dim), \
        f"CLS embedding shape should be ({batch_size}, {embedding_dim}), got {cls_embedding_eval.shape}"
    
    # Test 6: Verify CLS embedding is positioned at the end of the sequence
    assert transformer_out_forward.shape[0] == transformer_out_eval.shape[0], \
        "Sequence length should be the same in both forward and evaluate outputs"
    
    # Test 7: Verify that the CLS embedding is actually appended (sequence length increased by 1)
    original_seq_len = padded_emb.shape[0]
    output_seq_len = transformer_out_forward.shape[0]
    assert output_seq_len == original_seq_len + 1, \
        f"Output sequence length should be {original_seq_len + 1}, got {output_seq_len}"
    
    print(f"✓ CLS embedding consistency test passed")
    print(f"  - Forward CLS embedding shape: {cls_embedding_forward.shape}")
    print(f"  - Evaluate CLS embedding shape: {cls_embedding_eval.shape}")
    print(f"  - Sequence length: {original_seq_len} -> {output_seq_len}")


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
