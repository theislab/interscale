"""
Graph-level aggregation utilities for batched node-level tensors.

Used to convert node embeddings or node labels to one value per graph (e.g. before
a decoder or for graph-level ground truth).
"""
import torch


def aggregate_node_embeddings_to_graph(
    node_embedding: torch.Tensor,
    batch_ptr: torch.Tensor,
    reduce: str = "mean",
) -> torch.Tensor:
    """Aggregate node-level embeddings to one vector per graph.

    For converting node-level embeddings to graph-level, e.g. before a decoder
    for graph classification.

    Parameters
    ----------
    node_embedding : torch.Tensor
        Node embeddings, shape [N, E].
    batch_ptr : torch.Tensor
        PyG Batch.ptr, shape [B+1]; batch_ptr[b] and batch_ptr[b+1] are the
        start and end indices of graph b in the concatenated node dimension.
    reduce : str
        Reduction over nodes: "mean" (default).

    Returns
    -------
    torch.Tensor
        Graph-level embeddings, shape [B, E].
    """
    B = batch_ptr.shape[0] - 1
    device = node_embedding.device
    dtype = node_embedding.dtype
    E = node_embedding.shape[1]
    out = torch.zeros(B, E, device=device, dtype=dtype)
    for b in range(B):
        start = int(batch_ptr[b].item())
        end = int(batch_ptr[b + 1].item())
        chunk = node_embedding[start:end]
        if reduce == "mean":
            out[b] = chunk.mean(dim=0)
        else:
            raise ValueError(f"reduce must be 'mean', got {reduce!r}")
    return out


def aggregate_node_values_to_graph(
    node_values: torch.Tensor,
    batch_ptr: torch.Tensor,
    reduce: str = "mean",
) -> torch.Tensor:
    """Aggregate node-level values to one value per graph.

    Use for graph-level ground truth so all nodes in each graph contribute
    (e.g. mean of batch.y over nodes) instead of using only the first node.

    Parameters
    ----------
    node_values : torch.Tensor
        Node-level values, shape [N, ...] (e.g. batch.y [N, C] for one-hot).
    batch_ptr : torch.Tensor
        PyG Batch.ptr, shape [B+1].
    reduce : str
        Reduction over nodes: "mean" (default).

    Returns
    -------
    torch.Tensor
        Graph-level values, shape [B, ...].
    """
    B = batch_ptr.shape[0] - 1
    device = node_values.device
    dtype = node_values.dtype
    trailing = node_values.shape[1:]
    out = torch.zeros(B, *trailing, device=device, dtype=dtype)
    for b in range(B):
        start = int(batch_ptr[b].item())
        end = int(batch_ptr[b + 1].item())
        chunk = node_values[start:end]
        if reduce == "mean":
            out[b] = chunk.mean(dim=0)
        else:
            raise ValueError(f"reduce must be 'mean', got {reduce!r}")
    return out
