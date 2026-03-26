"""
Tests for graph-level label prediction with DualDecoderCombinedModuleClass.

Uses minimal in-test YACS config (no dependency on conftest package name).
"""
import pytest
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from yacs.config import CfgNode as CN
from torch import nn

from InterScale.module.combined_module.dual_decoder_combined_module import DualDecoderCombinedModuleClass


def get_dual_decoder_config(
    n_embed=4,
    max_seq_len=20,
    n_heads=2,
    dim_feedforward=64,
    num_layers=2,
    long_range_attention=False,
):
    """Build minimal YACS config for DualDecoderCombinedModuleClass.

    n_embed must be divisible by n_heads. Used so tests do not depend on
    conftest or external package config.
    """
    cfg = CN()
    cfg.model = CN()
    cfg.model.n_embed = n_embed
    cfg.model.decoder = CN()
    cfg.model.decoder.type = "linear"
    cfg.model.decoder.dropout_decoder = 0.1
    cfg.model.decoder.hidden_dims = [64, 32]

    cfg.model.local_component = CN()
    cfg.model.local_component.name = "GCN"
    cfg.model.local_component.parameters = CN()
    cfg.model.local_component.parameters.num_layers = num_layers
    cfg.model.local_component.parameters.hidden_dim = 64
    cfg.model.local_component.parameters.dropout_local = 0.1

    cfg.model.global_component = CN()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg.model.global_component.parameters = CN()
    cfg.model.global_component.parameters.max_seq_len = max_seq_len
    cfg.model.global_component.parameters.n_heads = n_heads
    cfg.model.global_component.parameters.dropout_global = 0.1
    cfg.model.global_component.parameters.activation_func = "relu"
    cfg.model.global_component.parameters.num_layers = num_layers
    cfg.model.global_component.parameters.dim_feedforward = dim_feedforward
    cfg.model.global_component.parameters.long_range_attention = long_range_attention
    cfg.model.global_component.parameters.type_gex_embedding = None

    return cfg


def create_graph_level_data(
    num_nodes,
    num_features,
    num_classes,
    same_input_per_node=False,
    seed=42,
):
    """One PyG Data graph with graph-level label (same one-hot y for all nodes)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if same_input_per_node:
        row = torch.randn(1, num_features)
        x = row.expand(num_nodes, num_features)
    else:
        x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor([[i, i + 1] for i in range(num_nodes - 1)], dtype=torch.long).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    y = torch.zeros(num_nodes, num_classes)
    class_idx = torch.randint(0, num_classes, (1,)).item()
    y[:, class_idx] = 1.0
    data = Data(x=x, edge_index=edge_index, y=y)
    data.obs_names = torch.arange(num_nodes)
    data.mask = torch.ones(num_nodes, dtype=torch.bool)
    return data


@pytest.fixture
def graph_level_batch():
    """Batch of 2 graphs with graph-level labels; dimensions aligned with config."""
    torch.manual_seed(42)
    np.random.seed(42)
    n_input, n_output = 5, 3
    graphs = [
        create_graph_level_data(6, n_input, n_output, same_input_per_node=False, seed=42 + i)
        for i in range(2)
    ]
    batch = Batch.from_data_list(graphs)
    return batch


@pytest.fixture
def graph_level_batch_same_input():
    """Two identical small graphs (2 nodes each, same features) so local/global outputs can satisfy graph-level assertions.

    Implementation expects B+2 masked nodes (B=num_graphs) and identical predictions per node/graph.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    n_input, n_output = 5, 3
    num_nodes_per_graph = 2
    num_graphs = 2
    graphs = []
    for _ in range(num_graphs):
        data = create_graph_level_data(
            num_nodes_per_graph, n_input, n_output, same_input_per_node=True, seed=42
        )
        graphs.append(data)
    batch = Batch.from_data_list(graphs)
    return batch


@pytest.fixture
def dual_decoder_module(graph_level_batch):
    """DualDecoderCombinedModule with minimal config; pct_mask_nodes=0 for graph-level stability."""
    cfg = get_dual_decoder_config()
    n_input = graph_level_batch.x.shape[1]
    n_output = graph_level_batch.y.shape[1]
    n_embed = 4
    module = DualDecoderCombinedModuleClass(
        cfg=cfg,
        n_input=n_input,
        n_output=n_output,
        n_embed=n_embed,
        decoder_type="linear",
        dropout_decoder=0.1,
        decoder_hidden_dims=[64, 32],
        pct_mask_nodes=0.0,
    )
    return module


def test_dual_decoder_forward_graph_batch(dual_decoder_module, graph_level_batch):
    """Forward pass on graph-level batch: no NaNs, expected shapes."""
    module = dual_decoder_module
    batch = graph_level_batch
    batch_masked, _ = module._common_step_masking(batch)

    local_embedding, global_embedding, src_padding_mask, pad_index_nodes, attention_mask, attn = module.forward(
        batch_masked
    )

    N = batch_masked.x.shape[0]
    B = int(batch_masked.batch.max().item()) + 1
    n_embed = module.n_embed

    assert not torch.any(torch.isnan(local_embedding)), "local_embedding contains NaN"
    assert not torch.any(torch.isnan(global_embedding)), "global_embedding contains NaN"
    assert local_embedding.shape == (N, n_embed)
    assert global_embedding.dim() == 3
    assert global_embedding.shape[1] == B
    assert global_embedding.shape[2] == n_embed
    assert src_padding_mask is not None
    assert pad_index_nodes is not None


@pytest.mark.parametrize("pct_mask_nodes", [0.0])
def test_dual_decoder_common_step_graph_classification(
    graph_level_batch_same_input, pct_mask_nodes
):
    """_common_step with prediction_task=classification, prediction_level=graph.

    Uses a single graph with same input per node so the local decoder can
    produce identical predictions per node (satisfies the graph-level assertion).
    """
    torch.manual_seed(42)
    np.random.seed(42)
    cfg = get_dual_decoder_config()
    batch = graph_level_batch_same_input
    n_input = batch.x.shape[1]
    n_output = batch.y.shape[1]
    n_embed = 4
    module = DualDecoderCombinedModuleClass(
        cfg=cfg,
        n_input=n_input,
        n_output=n_output,
        n_embed=n_embed,
        decoder_type="linear",
        dropout_decoder=0.1,
        decoder_hidden_dims=[64, 32],
        pct_mask_nodes=pct_mask_nodes,
    )
    num_graphs = int(batch.batch.max().item()) + 1

    local_emb, global_emb, y_pred_combined, y_true_combined, attn = module._common_step(
        batch, prediction_task="classification", prediction_level="graph"
    )

    assert not torch.any(torch.isnan(local_emb)), "local_embedding contains NaN"
    assert not torch.any(torch.isnan(global_emb)), "global_embedding contains NaN"
    assert not torch.any(torch.isnan(y_pred_combined)), "y_pred_combined contains NaN"
    assert not torch.any(torch.isnan(y_true_combined)), "y_true_combined contains NaN"
    assert len(y_pred_combined) == len(y_true_combined)
    assert len(y_true_combined) == 2 * num_graphs
    assert module._n_masked_nodes is not None
    assert module._is_graph_level is True


def test_dual_decoder_compute_separate_losses_graph_level(graph_level_batch_same_input):
    """compute_separate_losses after graph-level _common_step returns finite scalar losses."""
    torch.manual_seed(42)
    np.random.seed(42)
    cfg = get_dual_decoder_config()
    batch = graph_level_batch_same_input
    n_input = batch.x.shape[1]
    n_output = batch.y.shape[1]
    n_embed = 4
    module = DualDecoderCombinedModuleClass(
        cfg=cfg,
        n_input=n_input,
        n_output=n_output,
        n_embed=n_embed,
        decoder_type="linear",
        dropout_decoder=0.1,
        decoder_hidden_dims=[64, 32],
        pct_mask_nodes=0.0,
    )

    _, _, y_pred_combined, y_true_combined, _ = module._common_step(
        batch, prediction_task="classification", prediction_level="graph"
    )

    def loss_fn(pred, true):
        return nn.functional.cross_entropy(pred, true.argmax(dim=1))

    losses = module.compute_separate_losses(
        loss_fn, "CrossEntropy", y_pred_combined, y_true_combined
    )

    assert "local_loss" in losses
    assert "global_loss" in losses
    assert losses["local_loss"] is not None
    assert losses["global_loss"] is not None
    assert torch.isfinite(losses["local_loss"]).item()
    assert torch.isfinite(losses["global_loss"]).item()


if __name__ == "__main__":
    # Simple entry point to debug DualDecoderCombinedModuleClass with Cursor's debugger.
    torch.manual_seed(42)
    np.random.seed(42)

    # Build minimal config and a small graph-level batch
    cfg = get_dual_decoder_config()
    n_input, n_output = 5, 3
    num_nodes_per_graph = 2
    num_graphs = 2
    graphs = []
    for i in range(num_graphs):
        graphs.append(
            create_graph_level_data(
                num_nodes_per_graph,
                n_input,
                n_output,
                same_input_per_node=True,
                seed=42 + i,
            )
        )
    batch = Batch.from_data_list(graphs)

    n_embed = 4
    module = DualDecoderCombinedModuleClass(
        cfg=cfg,
        n_input=n_input,
        n_output=n_output,
        n_embed=n_embed,
        decoder_type="linear",
        dropout_decoder=0.1,
        decoder_hidden_dims=[64, 32],
        pct_mask_nodes=0.0,
    )

    # Place breakpoints in this block or inside dual_decoder_combined_module._common_step
    local_emb, global_emb, y_pred_combined, y_true_combined, attn = module._common_step(
        batch, prediction_task="classification", prediction_level="graph"
    )

    print("local_emb shape:", tuple(local_emb.shape))
    print("global_emb shape:", tuple(global_emb.shape))
    print("y_pred_combined shape:", tuple(y_pred_combined.shape))
    print("y_true_combined shape:", tuple(y_true_combined.shape))
