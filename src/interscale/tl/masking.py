import torch
from torch_geometric.data import Batch

MASK_VALUE = 0


def apply_mask(batched_data: Batch):
    """Mask nodes from PyG object in .mask attribute.

    Args:
        batched_data (Batch): _description_

    Returns
    -------
        batched_data_w_mask (Batch):
            Batch only containing nodes that were not masked
        mask_idx (torch.Tensor):
            Indices of masked nodes

    Example:
        Data object:
        x = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        data = Data(x=x, edge_index=edge_index, mask=mask)
        ----
        mask_idx = torch.tensor([1, 3])
        masked_values = torch.tensor([[0., 0.], [3., 4.], [0., 0.], [7., 8.]])
    """
    assert batched_data.mask is not None, "Mask is not set in the batch."

    mask = batched_data.mask
    mask_idx = torch.where(mask == 1)[0]  # TODO into 2D array [B, N_batched_nodes]
    masked_values = batched_data.x.clone()
    masked_values[mask] = MASK_VALUE
    batched_data_w_mask = batched_data.clone()
    batched_data_w_mask.x = masked_values
    return batched_data_w_mask, mask_idx


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
