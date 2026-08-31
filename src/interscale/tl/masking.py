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
        ``[N, G]`` boolean, or ``None``.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    if entry_mask is None:
        if loss_type == "GaussianNLL":
            return loss_fn(y_pred, y_true, torch.std(y_true, dim=1, keepdim=True))
        return loss_fn(y_pred, y_true)

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


def create_transformer_attention_mask_from_edges(
    edge_index: torch.Tensor, num_nodes: int, batch: torch.Tensor, index_nodes: list, num_heads: int
) -> torch.Tensor:
    """
    Creates an attention mask that is inverse to the edge indices. Unmasked = 0 and masked = -inf
    If two nodes are connected in the adjacency matrix (edge_index = 1) then we have no attention (0) and vice versa.

    Args:
        edge_index (torch.Tensor): Edge index tensor of shape [2, num_edges]
        num_nodes (int): Number of nodes in the graph
        batch (torch.Tensor): Batch tensor of shape [num_nodes]
        index_nodes (list): List of indices of nodes to keep [B, S] (range: 0, num_nodes)
        num_heads (int): Number of attention heads
    Returns:
        torch.Tensor: Attention mask of shape [num_batch*num_heads, max_seq_len, max_seq_len] with 1s for no attention (True -> mask attention) and 0s for attention (False -> no mask)
    """
    INVALID_MASK_VALUE = -float("inf")

    num_batch = int(batch[-1].item() + 1)
    max_seq_len = max(len(nodes) for nodes in index_nodes)

    # Initialize with -inf (no attention allowed)
    attention_mask = torch.full(
        (num_batch * num_heads, max_seq_len + 1, max_seq_len + 1), INVALID_MASK_VALUE, device=edge_index.device
    )
    # Set the diagonal to -inf (no self-attention)
    diag_idx = torch.arange(max_seq_len, device=edge_index.device)
    attention_mask[:, diag_idx, diag_idx] = INVALID_MASK_VALUE

    # Create full adjacency matrix + 1 for cls token (end of sequence)
    adj_matrix = torch.zeros((num_nodes, num_nodes), device=edge_index.device)  # TODO: check if zero or ones
    adj_matrix[edge_index[0], edge_index[1]] = INVALID_MASK_VALUE

    # For each batch, extract the submatrix for kept nodes
    for b in range(num_batch):
        nodes = index_nodes[b]
        seq_len = len(nodes)
        assert seq_len + 1 <= max_seq_len + 1, f"Mismatch: seq_len+1: {seq_len + 1}, max_seq_len+1: {max_seq_len + 1}"
        # Extract submatrix for the kept nodes
        batch_mask = adj_matrix[nodes][:, nodes]  # Get submatrix for kept nodes
        # INSERT_YOUR_CODE
        assert torch.any(batch_mask != 0), "batch_mask contains only zero entries"
        # Add row and column of ones for CLS token - full attention
        batch_mask = torch.cat(
            [batch_mask, torch.zeros(batch_mask.size(0), 1, device=batch_mask.device)], dim=1
        )  # Add column
        batch_mask = torch.cat(
            [batch_mask, torch.zeros(1, batch_mask.size(1), device=batch_mask.device)], dim=0
        )  # Add row
        assert batch_mask.shape == (seq_len + 1, seq_len + 1), (
            f"Mismatch: batch_mask.shape: {batch_mask.shape}, (seq_len+1, seq_len+1): {(seq_len + 1, seq_len + 1)}"
        )
        assert attention_mask.shape[-2:] == (max_seq_len + 1, max_seq_len + 1), (
            f"Mismatch: attention_mask.shape[-2:]: {attention_mask.shape[-2:]}, (seq_len+1, seq_len+1): {(seq_len + 1, seq_len + 1)}"
        )
        # append inverse adjacency matrix to the end of the attention mask
        attention_mask[b * num_heads : b * num_heads + num_heads, -(seq_len + 1) :, -(seq_len + 1) :] = batch_mask
        # add zeros for nodes that are not in the batch
        attention_mask[b * num_heads : b * num_heads + num_heads, :seq_len, :seq_len] = float("0")

    assert not torch.any(torch.isnan(attention_mask)), "attention_mask contains NaN values"
    print("attention_mask", attention_mask.shape, attention_mask)
    return attention_mask


def attn_mask_diagonal(batch: torch.Tensor, index_nodes: list, num_heads: int, device: torch.device) -> torch.Tensor:
    """
    Sets the diagonal of the attention mask to -inf.
    """
    max_seq_len = max(len(nodes) for nodes in index_nodes)
    batch_size = int(batch[-1].item() + 1)
    attention_mask = torch.zeros(
        (num_heads * batch_size, max_seq_len + 1, max_seq_len + 1), device=device, dtype=torch.float32
    )
    # Set the diagonal to -inf (no self-attention)
    diag_idx = torch.arange(max_seq_len, device=device)
    attention_mask[:, diag_idx, diag_idx] = float("-inf")
    # Convert attention_mask to same dtype as src_padding_mask
    return attention_mask
