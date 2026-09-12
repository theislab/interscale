"""Input corruption for the masked-reconstruction objective.

Two granularities are available, selected by ``mask_strategy``:

``"node"`` -- cell masking (the original behaviour)
    A Bernoulli subset of *cells* has its entire expression vector replaced by ``MASK_VALUE``,
    and the loss is evaluated on all G genes of those cells. A masked cell carries no
    information about itself, so the only thing the model can condition on is its neighbourhood,
    and E[x_i | neighbours of i] is close to the population mean. A near-constant predictor is
    therefore a strong solution to this objective, which is what makes it look like the model
    "learns the mean instead of reconstructing".

``"gene"`` -- per-entry masking (GraphMAE / MAE style)
    Every cell keeps most of its expression vector; a Bernoulli subset of *entries* ``(cell,
    gene)`` is replaced by ``MASK_VALUE``, drawn independently per cell. The loss is evaluated
    on those entries only (see :func:`masked_loss` below). The model now has the
    cell's remaining genes to condition on, so within-cell co-expression -- not just the
    population mean -- is available and rewarded.

Note on GraphMAE (arXiv:2205.10803): GraphMAE itself masks whole nodes, exactly as ``"node"``
does here. What it changes to avoid the trivial solution is the *criterion* (scaled cosine
error, already available as ``interscale.train.losses.SCELoss``), a learnable ``[MASK]`` token
rather than a zero vector, re-mask decoding, and a GNN decoder. Per-entry masking is the
orthogonal knob this module adds; the two are independent and can be ablated together.

THE FILL VALUE IS -1, NOT 0, AND IS NOT CONFIGURABLE. 62% of legnini23's ``log1p_norm`` entries
are exactly 0, and 648 of its cells are all-zero outright, so a zero fill makes a masked position
indistinguishable from a real measurement -- under gene masking the corruption becomes invisible
and the model cannot identify the entries it is being asked to reconstruct. ``MASK_VALUE = -1``
is outside the range of any log1p-normalised layer, so a masked position is unambiguous.

This was measured, not assumed (legnini23, gene masking at rate 0.25, 3 seeds):

    fill 0     val_concordance 0.0696 +/- 0.0113   (CV 16.2%)
    fill -1    val_concordance 0.0799 +/- 0.0046   (CV  5.8%)

-1 wins on every seed and cuts the run-to-run spread by ~2.5x, to below the cell-masking
baseline's own CV of 10.6%. A learnable per-gene [MASK] token (GraphMAE's design) was also tried
and scored identically to the fixed -1 to three decimals: the token drifts only ~0.02-0.035 over
100 epochs from any initialisation, so the fill behaves as a constant, not as something worth
learning. It was removed rather than kept as an option.
"""

import torch
from torch_geometric.data import Batch

# -1, not 0: outside the range of any log1p-normalised expression layer (which is >= 0), so a
# masked position can never be confused with a real measurement. See the module docstring for the
# measurement that settled this. Changing it back to 0 silently un-does that result.
MASK_VALUE = -1.0

MASK_STRATEGIES = ("node", "gene")


def sample_node_mask(num_nodes: int, pct: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw a per-cell mask: each cell is masked independently with probability ``pct``.

    Parameters
    ----------
    num_nodes
        Number of cells in the graph.
    pct
        Per-cell masking probability.
    generator
        Optional RNG, for reproducible draws.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape ``[num_nodes]``. At least one cell is always masked, otherwise
        the graph contributes no supervision at all.
    """
    mask = torch.rand(num_nodes, generator=generator) < pct
    if not mask.any():
        mask[torch.randint(num_nodes, (1,), generator=generator)] = True
    return mask


def sample_gene_mask(
    num_nodes: int, num_genes: int, pct: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Draw a per-entry mask: each ``(cell, gene)`` entry is masked independently with prob ``pct``.

    The draw is independent per cell, so different cells lose different genes. That is
    deliberate -- a mask shared across all cells of a graph would let the model learn a fixed
    "these G_masked genes are always missing" shortcut, and would make each step's supervision
    a single gene subset rather than |cells| different ones.

    Parameters
    ----------
    num_nodes
        Number of cells in the graph.
    num_genes
        Number of genes (columns of ``data.x``).
    pct
        Per-entry masking probability.
    generator
        Optional RNG, for reproducible draws.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape ``[num_nodes, num_genes]``. Every row has at least one masked
        entry, so every cell contributes to the loss and the per-cell cosine metric is defined
        for all of them.
    """
    mask = torch.rand(num_nodes, num_genes, generator=generator) < pct

    # Rows that came up all-False would silently drop out of the loss and make per-cell metrics
    # undefined; give each of them exactly one masked gene.
    empty_rows = ~mask.any(dim=1)
    if empty_rows.any():
        fill = torch.randint(num_genes, (int(empty_rows.sum()),), generator=generator)
        mask[empty_rows, fill] = True
    return mask


def apply_mask(batched_data: Batch, mask_strategy: str = "node"):
    """Corrupt ``batched_data.x`` at the granularity named by ``mask_strategy``.

    Under ``"gene"`` the batch's ``gene_mask`` ``[N, G]`` selects the entries to overwrite, and it
    is handed back so the loss can be restricted to them. Under ``"node"`` every gene of every
    cell selected by ``.mask`` is overwritten and the returned entry mask is ``None``, meaning
    "score the full rows".

    The strategy is a parameter rather than being inferred from whether a ``gene_mask`` attribute
    happens to be present. Sniffing the attribute made a stale ``gene_mask`` -- left on a ``Data``
    object reused across strategies -- silently override the configured strategy, which the
    dataloader then had to defend against by deleting the attribute.

    Args:
        batched_data (Batch): batch carrying ``.mask`` ``[N]``, plus ``.gene_mask`` ``[N, G]`` when
            ``mask_strategy == "gene"``.
        mask_strategy: one of :data:`MASK_STRATEGIES`.

    Returns
    -------
        batched_data_w_mask (Batch):
            Copy of the batch with the masked entries set to ``MASK_VALUE``.
        mask_idx (torch.Tensor):
            Indices of the cells that carry at least one masked entry -- i.e. the rows on which
            predictions are scored.
        entry_mask (torch.Tensor | None):
            ``[N, G]`` boolean over the *full* node ordering under gene masking, ``None`` under
            cell masking. Callers must subset it with the same indices they use for ``y_true``.

    Example:
        Data object:
        x = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        data = Data(x=x, edge_index=edge_index, mask=mask)
        ----
        mask_idx = torch.tensor([0, 2])
        masked_values = torch.tensor([[-1., -1.], [3., 4.], [-1., -1.], [7., 8.]])
    """
    assert batched_data.mask is not None, "Mask is not set in the batch."
    assert mask_strategy in MASK_STRATEGIES, f"mask_strategy must be one of {MASK_STRATEGIES}, got {mask_strategy!r}."

    gene_mask = getattr(batched_data, "gene_mask", None) if mask_strategy == "gene" else None
    masked_values = batched_data.x.clone()

    if gene_mask is None:
        assert mask_strategy == "node", "mask_strategy='gene' but the batch carries no gene_mask."
        mask = batched_data.mask
        mask_idx = torch.where(mask == 1)[0]  # TODO into 2D array [B, N_batched_nodes]
        masked_values[mask] = MASK_VALUE
        entry_mask = None
    else:
        gene_mask = gene_mask.bool()
        assert gene_mask.shape == batched_data.x.shape, (
            f"Mismatch: gene_mask.shape: {tuple(gene_mask.shape)}, x.shape: {tuple(batched_data.x.shape)}"
        )
        masked_values[gene_mask] = MASK_VALUE
        mask_idx = torch.where(gene_mask.any(dim=1))[0]
        entry_mask = gene_mask

    batched_data_w_mask = batched_data.clone()
    batched_data_w_mask.x = masked_values
    return batched_data_w_mask, mask_idx, entry_mask


# Losses whose value depends on the *arrangement* of a row, not just on the individual entries:
# they normalise or centre along dim=-1. Selecting entries out of them would change what a "row"
# is, so those get the masked entries zeroed in both tensors instead -- which restricts every sum,
# dot product and norm involved to the masked coordinates, leaving the row structure intact.
_ROW_STRUCTURED_LOSSES = ("SCELoss", "SCE_EntropyATT_Loss", "BalancedPearsonCorrelationLoss")


def masked_row_std(y: torch.Tensor, entry_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-row standard deviation of ``y`` over the masked entries only, shape ``[N, 1]``.

    Note: this is the *spread* argument the existing ``GaussianNLL`` branch passes, and it is
    deliberately std, not variance. ``nn.GaussianNLLLoss`` documents its third argument as a
    variance, and the pre-existing unmasked branch has always passed ``torch.std(...)`` there --
    a real bug, but one the published cell-masking runs were trained with. Matching it keeps the
    two masking arms comparable; fix both call sites together, never just this one, or the
    ablation stops being an ablation.
    """
    m = entry_mask.to(y.dtype)
    n = m.sum(dim=1, keepdim=True).clamp(min=1)
    mean = (y * m).sum(dim=1, keepdim=True) / n
    var = (((y * m) ** 2).sum(dim=1, keepdim=True) / n - mean**2).clamp(min=0)
    return var.sqrt().clamp(min=eps)


def _plain_loss(loss_fn, loss_type: str, y_pred: torch.Tensor, y_true: torch.Tensor):
    """``loss_fn`` over everything it is given, with GaussianNLL's third argument supplied."""
    if loss_type == "GaussianNLL":
        return loss_fn(y_pred, y_true, torch.std(y_true, dim=1, keepdim=True))
    return loss_fn(y_pred, y_true)


def masked_loss(loss_fn, loss_type: str, y_pred: torch.Tensor, y_true: torch.Tensor, entry_mask=None):
    """Evaluate a reconstruction loss on the masked entries only.

    Under cell masking (``entry_mask is None``) every entry of every scored row was blanked, so
    this is just ``loss_fn(y_pred, y_true)`` and the behaviour is unchanged. Under gene masking
    most entries of a scored row were *given to the model as input*; including them would let
    the identity map dominate the objective and would make the reported loss incomparable to the
    cell-masking arm.

    Parameters
    ----------
    loss_fn
        The configured criterion, e.g. ``nn.SmoothL1Loss()``.
    loss_type
        Its name as it appears in ``optim.loss``; selects how the restriction is applied.
    y_pred, y_true
        ``[N, G]`` predictions and targets for the scored rows.
    entry_mask
        ``[N, G]`` boolean, or ``None``. Under cell masking this is the per-cell mask broadcast
        over all genes, so it is all-True or all-False per row and the row branch below fires.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    if entry_mask is None:
        return _plain_loss(loss_fn, loss_type, y_pred, y_true)

    # Whole-row mask, i.e. cell masking expressed at entry level: every row is either entirely
    # in or entirely out. Subset the ROWS rather than the entries, which keeps each row intact --
    # a row-normalising criterion (SCE's cosine, Pearson) is only meaningful on a whole row, and
    # zeroing the excluded rows instead would feed it rows of zeros that dilute the mean.
    # This branch reproduces the pre-all-cells behaviour of cell masking exactly.
    rows_in = entry_mask.all(dim=1)
    if bool((rows_in | (~entry_mask).all(dim=1)).all()):
        return _plain_loss(loss_fn, loss_type, y_pred[rows_in], y_true[rows_in])

    if loss_type in _ROW_STRUCTURED_LOSSES:
        m = entry_mask.to(y_pred.dtype)
        return loss_fn(y_pred * m, y_true * m)

    if loss_type == "GaussianNLL":
        # std, matching the unmasked branch above -- see masked_row_std for why.
        spread = masked_row_std(y_true, entry_mask).expand_as(y_true)
        return loss_fn(y_pred[entry_mask], y_true[entry_mask], spread[entry_mask])

    # Element-wise criteria (MSELoss, SmoothL1, ...) reduce over whatever they are given, so
    # handing them the selected entries as a flat vector is exactly a mean over masked entries.
    return loss_fn(y_pred[entry_mask], y_true[entry_mask])


def _local_reach(edge_index: torch.Tensor, num_nodes: int, n_hops: int) -> torch.Tensor:
    """Boolean ``[num_nodes, num_nodes]`` matrix: which nodes the GNN can already see from each node.

    ``reach[i, j]`` is True when ``j`` lies within ``n_hops`` of ``i``, the diagonal included --
    a cell is part of its own receptive field. ``n_hops`` should be the number of message-passing
    layers of the local component, since that is exactly its receptive field.

    The multi-hop closure is done with sparse matmuls, so the cost tracks the number of edges
    rather than ``num_nodes ** 2``; only the final densification is quadratic, and that is
    unavoidable because the attention mask itself is dense.
    """
    device = edge_index.device

    if n_hops <= 1:
        # No closure to compute, so skip the sparse round trip: scattering straight into the
        # boolean matrix avoids materialising an n x n float one just to threshold it.
        dense = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
        dense[edge_index[0], edge_index[1]] = True
    else:
        values = torch.ones(edge_index.shape[1], device=device)
        adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).coalesce()

        reach, frontier = adj, adj
        for _ in range(n_hops - 1):
            frontier = torch.sparse.mm(frontier, adj).coalesce()
            reach = (reach + frontier).coalesce()
        dense = reach.to_dense() > 0
    # Spatial neighbour graphs are built symmetric, but a directed edge_index would otherwise
    # leave the mask asymmetric and block only one direction of a pair the GNN mixed both ways.
    # `dense |= dense.T` aliases its own memory, so this is an out-of-place or.
    dense = dense | dense.transpose(0, 1)
    dense.fill_diagonal_(True)
    return dense


def create_transformer_attention_mask_from_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    batch: torch.Tensor,
    index_nodes: list,
    num_heads: int,
    *,
    n_hops: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Block the transformer from attending inside the local component's receptive field.

    This is the ``M = 1 - A`` mask of the paper, with ``A`` taken as the ``n_hops`` closure of the
    spatial neighbour graph rather than just its direct edges: the point is to stop the two
    components from re-deriving the same signal, and a 2-layer GCN has already mixed the 2-hop
    neighbourhood. The diagonal is blocked with it -- a cell's own state is what the local
    embedding is.

    **Why this cannot produce NaN.** A softmax row that is entirely ``-inf`` is NaN, which is the
    failure this mask invites: a cell in a dense region can easily have every other cell of its
    window inside its own neighbourhood. Two properties rule it out here.

    * The CLS token is never blocked, as a query or as a key. Every row therefore keeps at least
      one attendable key, whatever the graph looks like. It is also never a padded key, so the
      merge with ``src_key_padding_mask`` cannot take that guarantee away.
    * The mask is boolean and is built by indexing, never by arithmetic. Forming it as
      ``(1 - A) * -inf`` -- the obvious reading of "inverse adjacency" -- puts ``0 * -inf`` on
      every connected pair, and that is NaN before the softmax ever runs.

    Padding positions are left unblocked for the same reason: their rows are meaningless but must
    still normalise, and ``src_key_padding_mask`` is what actually removes them as keys.

    Parameters
    ----------
    edge_index
        ``[2, num_edges]`` edge index of the whole batch, with PyG's per-graph node offsets.
    num_nodes
        Number of nodes in the batch; used to check ``batch`` and ``edge_index`` agree.
    batch
        ``[num_nodes]`` graph assignment per node.
    index_nodes
        Per graph, the indices of the nodes ``pad_batch`` kept, relative to that graph's own node
        order. Its lengths define the sequence length.
    num_heads
        Number of attention heads; the mask is repeated for each.
    n_hops
        Radius of the blocked neighbourhood, in message-passing steps. Pass the local component's
        ``num_layers``.
    device
        Device for the returned mask. Defaults to ``edge_index``'s.

    Returns
    -------
    torch.Tensor
        Boolean ``[num_batch * num_heads, S + 1, S + 1]`` mask, ``True`` where attention is
        blocked, with the CLS token in the last position. Ordered graph-major, matching what
        :class:`torch.nn.MultiheadAttention` expects of a 3-D ``attn_mask``.
    """
    device = edge_index.device if device is None else device
    batch = batch.to(torch.long)
    if batch.numel() != num_nodes:
        raise ValueError(f"batch has {batch.numel()} entries but num_nodes is {num_nodes}")

    num_batch = int(batch.max().item()) + 1
    max_seq_len = max(len(nodes) for nodes in index_nodes)
    mask = torch.zeros((num_batch, max_seq_len + 1, max_seq_len + 1), dtype=torch.bool, device=device)

    for b in range(num_batch):
        nodes_b = torch.nonzero(batch == b, as_tuple=False).flatten()
        n_b = int(nodes_b.numel())
        if n_b == 0:
            continue
        # PyG batches graphs by concatenation, so a graph's nodes are contiguous and its local
        # indices are the global ones minus the offset of its first node.
        offset = int(nodes_b[0].item())
        in_graph = (batch[edge_index[0]] == b) & (batch[edge_index[1]] == b)
        local_edges = edge_index[:, in_graph] - offset

        reach = _local_reach(local_edges, n_b, n_hops)

        kept = torch.as_tensor(index_nodes[b], dtype=torch.long, device=reach.device)
        block = reach[kept][:, kept].to(device)

        # pad_batch left-pads, so the kept tokens sit in the LAST len(kept) positions before the
        # CLS slot. Writing the block anywhere else silently masks the wrong pairs.
        s = int(kept.numel())
        lo = max_seq_len - s
        mask[b, lo:max_seq_len, lo:max_seq_len] = block

    if bool(mask.all(dim=-1).any()):
        raise RuntimeError("a query row is fully blocked; softmax would be NaN")

    # (N * num_heads, L, S) is indexed graph-major: repeat_interleave, not repeat.
    return mask.repeat_interleave(num_heads, dim=0)


def attn_mask_diagonal(batch: torch.Tensor, index_nodes: list, num_heads: int, device: torch.device) -> torch.Tensor:
    """Block self-attention only: the weakest mask, and the default.

    Returns the same boolean convention as
    :func:`create_transformer_attention_mask_from_edges` (``True`` = blocked), so the two are
    interchangeable at the call site. The CLS slot in the last position is left open.
    """
    max_seq_len = max(len(nodes) for nodes in index_nodes)
    batch_size = int(batch.max().item()) + 1
    attention_mask = torch.zeros(
        (num_heads * batch_size, max_seq_len + 1, max_seq_len + 1), device=device, dtype=torch.bool
    )
    diag_idx = torch.arange(max_seq_len, device=device)
    attention_mask[:, diag_idx, diag_idx] = True
    return attention_mask
