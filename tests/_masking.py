import pytest
import torch
from torch_geometric.data import Data, Batch
import numpy as np

from graph_transformer_long_range_niches.model.gnn_transformer import LitGNNTransformer
from omegaconf import OmegaConf

class TestTransformerMasking:
    @pytest.fixture
    def sample_config(self):
        cfg = {
            'dataset': {
                'num_features': 10,
                'num_classes': 5,
                'batch_size': 2,
                'prediction_task': 'node_classification'
            },
            'gnn': {
                'embed_dim': 32,
                'num_layers': 2
            },
            'transformer': {
                'd_model': 32,
                'max_seq_len': 5,
                'num_layers': 2,
                'nhead': 4
            },
            'optim': {
                'lr': 0.001,
                'wd': 0.01,
                'loss': 'CrossEntropy',
                'warm_up': 10
            }
        }
        return OmegaConf.create(cfg)

    @pytest.fixture
    def sample_batch(self):
        # Create two small graphs
        x1 = torch.randn(3, 10)  # 3 nodes, 10 features
        x2 = torch.randn(2, 10)  # 2 nodes, 10 features
        edge_index1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        edge_index2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        y1 = torch.eye(5)[torch.randint(0, 5, (3,))]  # One-hot encoded labels
        y2 = torch.eye(5)[torch.randint(0, 5, (2,))]

        data_list = [
            Data(x=x1, edge_index=edge_index1, y=y1),
            Data(x=x2, edge_index=edge_index2, y=y2)
        ]
        return Batch.from_data_list(data_list)

    def test_mask_nodes(self, sample_config, sample_batch):
        model = LitGNNTransformer(sample_config)
        
        # Test masking
        x_masked, masked_indices, original_values = model.mask_nodes(sample_batch.x, sample_batch.batch)
        
        # Basic checks
        assert x_masked.shape == sample_batch.x.shape
        assert len(masked_indices) == 2  # One node per graph (mask_ratio=1)
        assert len(original_values) == 2  # Original values for masked nodes
        
        # Check that masked values are different from original
        for idx in masked_indices:
            assert not torch.allclose(x_masked[idx], sample_batch.x[idx])
            
        # Check that mask token was used
        for idx in masked_indices:
            assert torch.allclose(x_masked[idx], model.mask_token.squeeze())

    def test_forward_masking(self, sample_config, sample_batch):
        model = LitGNNTransformer(sample_config)
        
        # Run forward pass
        z, out, masked_indices, original_values = model.forward(sample_batch)
        
        # Check output dimensions
        assert out.shape[0] == len(masked_indices)  # One prediction per masked node
        assert out.shape[1] == sample_config.dataset.num_classes
        
        # Check that we're getting predictions for masked nodes
        assert len(masked_indices) == 2  # One per graph
        assert len(original_values) == 2

    def test_different_mask_ratios(self, sample_config, sample_batch):
        # Test with different mask ratios
        mask_ratios = [1, 2]
        
        for ratio in mask_ratios:
            sample_config.mask_ratio = ratio
            model = LitGNNTransformer(sample_config)
            
            x_masked, masked_indices, original_values = model.mask_nodes(sample_batch.x, sample_batch.batch)
            
            # Count masks per graph
            graph_0_masks = (sample_batch.batch[masked_indices] == 0).sum()
            graph_1_masks = (sample_batch.batch[masked_indices] == 1).sum()
            
            print(masked_indices)
            
            # Check mask counts
            assert graph_0_masks <= min(ratio, 3)  # Graph 0 has 3 nodes
            assert graph_1_masks <= min(ratio, 2)  # Graph 1 has 2 nodes

    def test_training_step(self, sample_config, sample_batch):
        model = LitGNNTransformer(sample_config)
        
        # Run training step
        loss = model.training_step(sample_batch, 0)
        
        # Check that loss is computed and has gradient
        assert loss is not None
        assert isinstance(loss.item(), float)
        assert not torch.isnan(loss)
        assert loss.requires_grad

    def test_masking_consistency(self, sample_config, sample_batch):
        model = LitGNNTransformer(sample_config)
        
        # Run multiple forward passes
        results = [model.forward(sample_batch) for _ in range(5)]
        
        # Check that different nodes are being masked in different forward passes
        masked_indices_list = [r[2] for r in results]
        unique_masks = set(tuple(m.tolist()) for m in masked_indices_list)
        
        # It's highly unlikely to get the same masks 5 times in a row
        assert len(unique_masks) > 1, "Masking should be random across forward passes"