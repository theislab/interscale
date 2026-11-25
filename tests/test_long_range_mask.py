"""The long-range attention mask must block the GNN's receptive field without producing NaN.

`create_transformer_attention_mask_from_edges` implements `M = 1 - A`: the transformer may not
attend inside the neighbourhood the local component has already mixed. The failure this invites is
a softmax row that is entirely `-inf`, which is NaN -- a cell in a dense region can have every
other cell of its window inside its own neighbourhood. The guarantee that rules it out is that the
CLS token is never blocked as a key, so every row keeps at least one attendable position.

The other thing these tests pin down is index alignment. `pad_batch` *left*-pads, so a graph's
tokens occupy the last positions of the sequence; writing the adjacency block anywhere else masks
the wrong pairs silently, with no error and a model that still trains.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from interscale.module.global_modules import TransformerNodeEncoderHook
from interscale.tl.masking import attn_mask_diagonal, create_transformer_attention_mask_from_edges

N_HEADS = 2


def path_graph(n):
    """0-1-2-...-(n-1), as an undirected edge index."""
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def test_one_hop_blocks_neighbours_and_self():
    n = 5
    mask = create_transformer_attention_mask_from_edges(
        path_graph(n), n, torch.zeros(n, dtype=torch.long), [list(range(n))], num_heads=1, n_hops=1
    )
    blocked = mask[0, :n, :n]

    expected = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        expected[i, i] = True
        if i > 0:
            expected[i, i - 1] = True
        if i < n - 1:
            expected[i, i + 1] = True

    assert torch.equal(blocked, expected)


def test_two_hops_blocks_the_gcn_receptive_field():
    """A 2-layer GCN mixes the 2-hop neighbourhood, so blocking only 1 hop leaves the second hop
    reachable by both components -- the duplication the mask exists to prevent."""
    n = 7
    mask = create_transformer_attention_mask_from_edges(
        path_graph(n), n, torch.zeros(n, dtype=torch.long), [list(range(n))], num_heads=1, n_hops=2
    )
    blocked = mask[0, :n, :n]

    for i in range(n):
        for j in range(n):
            assert bool(blocked[i, j]) == (abs(i - j) <= 2), f"({i},{j}) at 2 hops"


def test_cls_token_is_never_blocked():
    """The NaN guarantee: whatever the graph, the CLS column stays open, so no row is empty."""
    n = 6
    dense = torch.tensor([[i, j] for i in range(n) for j in range(n) if i != j], dtype=torch.long).T
    mask = create_transformer_attention_mask_from_edges(
        dense, n, torch.zeros(n, dtype=torch.long), [list(range(n))], num_heads=1, n_hops=3
    )

    assert bool(mask[0, :n, :n].all()), "every real pair should be blocked in a complete graph"
    assert not bool(mask[0, -1, :].any()), "CLS row blocked"
    assert not bool(mask[0, :, -1].any()), "CLS column blocked"
    assert not bool(mask.all(dim=-1).any()), "a fully blocked row would be NaN after softmax"


def test_block_is_written_to_the_left_padded_positions():
    """Two graphs of different size: the smaller one's block must land at the END of its sequence,
    because that is where `pad_batch` puts its tokens."""
    small, large = 3, 5
    edges_small = path_graph(small)
    edges_large = path_graph(large) + small  # PyG offsets the second graph's node ids
    edge_index = torch.cat([edges_small, edges_large], dim=1)
    batch = torch.tensor([0] * small + [1] * large, dtype=torch.long)

    mask = create_transformer_attention_mask_from_edges(
        edge_index, small + large, batch, [list(range(small)), list(range(large))], num_heads=1, n_hops=1
    )

    pad = large - small
    assert not bool(mask[0, :pad, :].any()), "padded rows must stay open"
    assert not bool(mask[0, :, :pad].any()), "padded columns must stay open"
    # The 3-node path, placed in the last three real positions.
    assert torch.equal(
        mask[0, pad:large, pad:large],
        torch.tensor([[True, True, False], [True, True, True], [False, True, True]]),
    )


def test_heads_are_repeated_graph_major():
    """MultiheadAttention indexes a 3-D attn_mask as batch * num_heads + head, so the graphs must
    be interleaved by head and not tiled."""
    small, large = 2, 4
    edge_index = torch.cat([path_graph(small), path_graph(large) + small], dim=1)
    batch = torch.tensor([0] * small + [1] * large, dtype=torch.long)

    mask = create_transformer_attention_mask_from_edges(
        edge_index, small + large, batch, [list(range(small)), list(range(large))], num_heads=N_HEADS, n_hops=1
    )

    assert mask.shape == (2 * N_HEADS, large + 1, large + 1)
    assert torch.equal(mask[0], mask[1]), "both heads of graph 0"
    assert torch.equal(mask[2], mask[3]), "both heads of graph 1"
    assert not torch.equal(mask[0], mask[2]), "the two graphs differ"


def test_only_the_kept_nodes_are_used():
    """When a graph is longer than max_seq_len, pad_batch keeps a subset; the mask has to be the
    submatrix over exactly those nodes, in their order."""
    n = 6
    kept = [0, 2, 4]
    mask = create_transformer_attention_mask_from_edges(
        path_graph(n), n, torch.zeros(n, dtype=torch.long), [kept], num_heads=1, n_hops=1
    )
    blocked = mask[0, : len(kept), : len(kept)]

    # 0-2-4 are pairwise 2 apart on the path, so at 1 hop only the diagonal is blocked.
    assert torch.equal(blocked, torch.eye(len(kept), dtype=torch.bool))


def test_diagonal_mask_uses_the_same_boolean_convention():
    n = 4
    diag = attn_mask_diagonal(
        torch.zeros(n, dtype=torch.long), [list(range(n))], num_heads=1, device=torch.device("cpu")
    )

    assert diag.dtype == torch.bool
    assert torch.equal(diag[0, :n, :n], torch.eye(n, dtype=torch.bool))
    assert not bool(diag[0, -1, :].any()) and not bool(diag[0, :, -1].any())


class _Batch:
    """Minimal stand-in for the PyG batch the global module reads."""

    def __init__(self, edge_index, batch, n_nodes):
        self.edge_index = edge_index
        self.batch = batch
        self.obs_names = torch.arange(n_nodes)
        self.num_nodes = n_nodes
        self.mask = torch.zeros(n_nodes, dtype=torch.bool)


def build_module(long_range, hops, max_seq_len=32):
    return TransformerNodeEncoderHook(
        max_seq_len=max_seq_len,
        n_heads=N_HEADS,
        dropout_global=0.0,
        act_func="relu",
        num_layers=1,
        dim_feedforward=16,
        long_range_attention=long_range,
        local_mask_hops=hops,
        n_input=8,
        n_output=8,
        n_embed=8,
        decoder_type="linear",
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=0.1,
        mask_strategy="node",
    )


@pytest.mark.parametrize("hops", [1, 2, 3])
def test_forward_is_finite_on_a_dense_graph(hops):
    """The regression test for the NaN reports: a graph dense enough that many cells have their
    whole window inside their own neighbourhood still has to produce finite attention."""
    n = 12
    rng = np.random.default_rng(0)
    edge_index = torch.tensor([[i, j] for i in range(n) for j in range(n) if i != j], dtype=torch.long).T
    batch = _Batch(edge_index, torch.zeros(n, dtype=torch.long), n)
    emb = torch.tensor(rng.normal(size=(n, 8)), dtype=torch.float32)

    module = build_module(long_range=True, hops=hops).eval()
    padded, padding_mask, _, attn_mask = module.common_step_local_to_global(batch, emb, eval_step=True)
    out, _, attn = module.forward(padded, padding_mask, attn_mask, register_hook=True)

    assert torch.isfinite(out).all(), "transformer output contains NaN or inf"
    assert attn is not None and torch.isfinite(attn).all(), "attention weights contain NaN or inf"

    # A masked softmax can be finite forward and still produce NaN gradients, which is how this
    # fails silently in training rather than at the first batch.
    out.sum().backward()
    for name, param in module.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"non-finite gradient in {name}"


def test_training_steps_stay_finite():
    """Several optimiser steps under the mask, since a NaN that only appears once weights have
    moved would not be caught by a single forward."""
    n = 10
    rng = np.random.default_rng(2)
    edge_index = torch.tensor([[i, j] for i in range(n) for j in range(n) if i != j], dtype=torch.long).T
    batch = _Batch(edge_index, torch.zeros(n, dtype=torch.long), n)
    emb = torch.tensor(rng.normal(size=(n, 8)), dtype=torch.float32)
    target = torch.tensor(rng.normal(size=(n + 1, 1, 8)), dtype=torch.float32)

    module = build_module(long_range=True, hops=2)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-2)

    for _ in range(5):
        optimizer.zero_grad()
        padded, padding_mask, _, attn_mask = module.common_step_local_to_global(batch, emb, eval_step=True)
        out, _, _ = module.forward(padded, padding_mask, attn_mask, register_hook=False)
        loss = torch.nn.functional.mse_loss(out, target)
        assert torch.isfinite(loss), "loss went non-finite under the mask"
        loss.backward()
        optimizer.step()


def test_blocked_pairs_receive_no_attention():
    """The mask has to actually reach the softmax, not just be built."""
    n = 8
    rng = np.random.default_rng(1)
    edge_index = path_graph(n)
    batch = _Batch(edge_index, torch.zeros(n, dtype=torch.long), n)
    emb = torch.tensor(rng.normal(size=(n, 8)), dtype=torch.float32)

    module = build_module(long_range=True, hops=1).eval()
    padded, padding_mask, _, attn_mask = module.common_step_local_to_global(batch, emb, eval_step=True)
    module.forward(padded, padding_mask, attn_mask, register_hook=True)

    weights = module.transformer_encoder.layers[0].get_attn_output_weights()  # [B, H, L, S]
    weights = weights.reshape(-1, weights.shape[-2], weights.shape[-1])[0]
    blocked = attn_mask[0]

    assert torch.allclose(weights[blocked], torch.zeros(int(blocked.sum())), atol=1e-6)
    assert weights[~blocked].sum() > 0
    assert torch.allclose(weights.sum(dim=-1), torch.ones(weights.shape[0]), atol=1e-5)


def test_mask_off_leaves_everything_but_the_diagonal_open():
    n = 6
    batch = _Batch(path_graph(n), torch.zeros(n, dtype=torch.long), n)
    emb = torch.zeros(n, 8)

    module = build_module(long_range=False, hops=2).eval()
    _, _, _, attn_mask = module.common_step_local_to_global(batch, emb, eval_step=True)

    assert torch.equal(attn_mask[0, :n, :n], torch.eye(n, dtype=torch.bool))


def random_geometric_graph(n, radius, seed=0):
    """Irregular degrees, unlike the path graph: closer to a real spatial neighbour graph."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 1, size=(n, 2))
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j and np.linalg.norm(pos[i] - pos[j]) <= radius:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


@pytest.mark.parametrize("cls_name", ["GCN", "GIN"])
@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_mask_covers_exactly_the_local_receptive_field(cls_name, n_layers):
    """The mask must block every cell that actually reached the local embedding, and no more.

    Rather than asserting the hop arithmetic a second time, this measures the local module's real
    receptive field by autograd: cell j influenced cell i's local embedding exactly when
    d h_local[i] / d x[j] is non-zero. That catches anything the hop count would miss -- an
    off-by-one in the layer stack, self-loops, the input_proj residual.

    The mask is built from the module's own ``receptive_field_hops``, the number the combined
    modules hand the transformer, so this pins that property against the measured field too.

    The probe is a random projection of h[i], not h[i].sum(): the GCN ends in a LayerNorm with
    elementwise_affine=False, so every row sums to exactly zero and the gradient of the sum is
    identically zero regardless of the graph.
    """
    from interscale.module import local_modules

    n, n_features, n_embed = 12, 6, 4
    edge_index = random_geometric_graph(n, radius=0.35)

    torch.manual_seed(0)
    gcn = getattr(local_modules, cls_name)(
        n_layers=n_layers,
        hidden_dim=8,
        dropout_local=0.0,
        n_input=n_features,
        n_output=n_features,
        n_embed=n_embed,
        decoder_type=None,
        dropout_decoder=0.0,
        mask_percentage=0.1,
        mask_strategy="node",
    ).eval()
    probe = torch.randn(n_embed)

    # The receptive field is every j whose input CAN move h[i], so the reading is a union over a
    # few random inputs. One input is not enough for GIN: its ReLU MLPs can zero a cell's whole
    # gradient for one particular x, which reads as no influence at all.
    influenced = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        for _ in range(3):
            x = torch.randn(n, n_features, requires_grad=True)
            (gcn(x, edge_index)[i] * probe).sum().backward()
            influenced[i] |= x.grad.abs().sum(1) > 1e-10

    # Check the instrument before trusting its reading. A cell always influences its own
    # embedding -- through input_proj if through nothing else -- so an empty diagonal means the
    # gradient is dead and every comparison below is vacuous rather than informative. This is
    # exactly what a probe of h[i].sum() produces, and without this guard it surfaces as
    # "mask and receptive field disagree", pointing at the mask instead of at the measurement.
    assert bool(influenced.diagonal().all()), (
        "probe registered no self-influence: the gradient is dead, so this test measures nothing"
    )

    assert gcn.receptive_field_hops == n_layers
    mask = create_transformer_attention_mask_from_edges(
        edge_index,
        n,
        torch.zeros(n, dtype=torch.long),
        [list(range(n))],
        num_heads=1,
        n_hops=gcn.receptive_field_hops,
    )
    blocked = mask[0, :n, :n]

    leaked = influenced & ~blocked
    assert not bool(leaked.any()), (
        f"{int(leaked.sum())} pairs reached the local embedding but are left open to attention"
    )
    assert torch.equal(influenced, blocked), "mask and GCN receptive field disagree"


def test_paths_through_dropped_nodes_are_still_blocked():
    """Two kept cells that are only multi-hop neighbours *through* a cell pad_batch dropped were
    still mixed by the GCN, which ran on the whole graph before any token was dropped. The reach
    has to be closed on the full graph and only then restricted to the kept nodes."""
    n = 5
    kept = [0, 1, 3, 4]  # node 2 dropped; 1 and 3 are 2 hops apart only via node 2
    mask = create_transformer_attention_mask_from_edges(
        path_graph(n), n, torch.zeros(n, dtype=torch.long), [kept], num_heads=1, n_hops=2
    )

    pos_of = {node: t for t, node in enumerate(kept)}
    assert bool(mask[0, pos_of[1], pos_of[3]]), "2-hop pair via a dropped node left open"
    assert bool(mask[0, pos_of[3], pos_of[1]])
    assert not bool(mask[0, pos_of[0], pos_of[4]]), "0 and 4 are 4 hops apart and must stay open"


# --------------------------------------------------------------------------- the built model's reach
#
# Everything below checks the module a model actually builds. The reach used to be derived by a
# helper that was tested in isolation, while every real model built its transformer through a path
# that read a config key that did not exist -- and silently masked one hop.


def _component_cfg(local_name, *, n_layers=2, long_range=True):
    """A config naming both components, with the component schemas loaded as load_config does."""
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.config.local_component_config import get_local_component_cfg

    cfg = get_cfg_defaults()
    cfg.model.local_component.name = local_name
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_local_component_cfg(cfg, local_name)
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.local_component.parameters.num_layers = n_layers
    cfg.model.global_component.parameters.long_range_attention = long_range
    return cfg


def _combined(cfg, dual=False):
    from interscale.module.combined_module import CombinedModule, DualDecoderCombinedModule

    return (DualDecoderCombinedModule if dual else CombinedModule)(
        cfg=cfg,
        n_input=8,
        n_output=8,
        n_embed=8,
        decoder_type=None,
        dropout_decoder=0.0,
        decoder_hidden_dims=[16],
        mask_percentage=0.1,
        mask_strategy="node",
    )


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("local_name", ["GCN", "GIN"])
@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_the_built_transformer_masks_what_its_local_module_mixed(local_name, n_layers, dual):
    module = _combined(_component_cfg(local_name, n_layers=n_layers), dual=dual)

    assert module.local_module.receptive_field_hops == n_layers
    assert module.global_module.local_mask_hops == n_layers


def test_a_combined_forward_blocks_the_two_hop_closure_of_a_two_layer_gcn():
    """Through the module's own input step: on a path graph a 2-layer GCN mixed every pair at most
    two apart, so exactly those pairs are blocked -- not just the direct neighbours."""
    n = 8
    module = _combined(_component_cfg("GCN", n_layers=2))
    batch = _Batch(path_graph(n), torch.zeros(n, dtype=torch.long), n)

    _, _, _, mask = module.global_module.common_step_local_to_global(batch, torch.randn(n, 8), eval_step=True)

    blocked = mask[0, :n, :n]
    for i in range(n):
        for j in range(n):
            assert bool(blocked[i, j]) == (abs(i - j) <= 2), f"({i},{j})"


def test_a_local_module_without_message_passing_has_no_reach():
    """SCVI's config carries a num_layers too -- its encoder depth -- but it never looks at a
    neighbour. Reading the config called that two hops of graph mixing."""
    with pytest.warns(UserWarning, match="nothing to block"):
        module = _combined(_component_cfg("SCVI", n_layers=2))

    assert module.local_module.receptive_field_hops == 0
    assert module.global_module.local_mask_hops == 0


def test_reach_zero_is_the_self_only_mask():
    """Nothing mixed means nothing to block: long-range masking at reach 0 is exactly the default
    mask, and says so rather than silently doing nothing."""
    n = 6
    with pytest.warns(UserWarning, match="nothing to block"):
        module = build_module(long_range=True, hops=0).eval()
    batch = _Batch(path_graph(n), torch.zeros(n, dtype=torch.long), n)

    _, _, index_nodes, mask = module.common_step_local_to_global(batch, torch.randn(n, 8), eval_step=True)

    assert torch.equal(mask, attn_mask_diagonal(batch.batch, index_nodes, N_HEADS, mask.device))


def test_the_mask_builder_refuses_a_reach_of_zero():
    """`_local_reach` reads 0 as 1, so a zero reach must never get that far."""
    with pytest.raises(ValueError, match="n_hops must be >= 1"):
        create_transformer_attention_mask_from_edges(
            path_graph(4), 4, torch.zeros(4, dtype=torch.long), [list(range(4))], num_heads=1, n_hops=0
        )


def test_from_config_cannot_be_called_without_the_reach():
    """A forgotten argument must fail rather than default: a default is how every run got one hop."""
    from interscale.module.base import GlobalModule

    with pytest.raises(TypeError, match="local_mask_hops"):
        GlobalModule.from_config(_component_cfg("GCN"), n_input=8, n_output=8, n_embed=8)


def test_the_config_has_no_hop_count_to_set(tmp_path):
    """The reach follows the local module. A yaml still setting the removed knob must fail to load
    rather than be silently ignored."""
    from interscale.config import load_config

    path = tmp_path / "old.yaml"
    path.write_text(
        "model:\n  global_component:\n    name: self-attn-transformer\n    parameters:\n      long_range_mask_hops: 2\n"
    )
    with pytest.raises(KeyError, match="long_range_mask_hops"):
        load_config(path)


def _setup_adata(model_cls):
    import pandas as pd
    from anndata import AnnData

    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.integers(0, 10, size=(6, 4)).astype(np.float32))
    adata.obs["sample"] = pd.Categorical(["s1"] * 3 + ["s2"] * 3)
    adata.obs["split"] = pd.Categorical(["train"] * 4 + ["val"] * 2)
    model_cls._setup_anndata(
        adata=adata, layer_key=None, sample_key_list=["sample"], prediction_task="regression", view_registry=False
    )
    return adata


def test_a_combined_model_masks_its_gcn_depth():
    """The class a user instantiates, end to end: the default 2-layer GCN gives a 2-hop mask."""
    from interscale.model.combined_model import CombinedModel

    model = CombinedModel(_setup_adata(CombinedModel), cfg=_component_cfg("GCN", n_layers=2))

    assert model.module.global_module.local_mask_hops == 2


def test_a_global_model_has_nothing_to_mask_and_says_so():
    """No local component, so nothing was mixed: reach 0, and long-range masking warns that it
    does nothing instead of blocking a neighbourhood no component aggregated."""
    from interscale.config import get_cfg_defaults
    from interscale.config.global_component_config import get_global_component_cfg
    from interscale.model.global_model import GlobalModel

    cfg = get_cfg_defaults()
    cfg.model.global_component.name = "self-attn-transformer"
    cfg = get_global_component_cfg(cfg, "self-attn-transformer")
    cfg.model.global_component.parameters.long_range_attention = True

    with pytest.warns(UserWarning, match="nothing to block"):
        model = GlobalModel(_setup_adata(GlobalModel), cfg=cfg)

    assert model.module.local_mask_hops == 0
