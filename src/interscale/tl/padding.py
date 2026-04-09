import random

import torch

"""Utility functions for padding input embedding to transformer"""


def _select_masked_nodes(keep_indices, mask, num_nodes_i, max_seq_len):
    """Case 1.1: Node masking of input data then select masked nodes to keep."""
    must_keep = keep_indices[mask]  # torch.tensor(Bool) [G]
    other_nodes = ~must_keep  # torch.tensor(Bool)

    # Get indices of must-keep nodes and other nodes separately
    keep_idx = torch.where(must_keep)[0].tolist()
    other_idx = torch.where(other_nodes)[0].tolist()
    assert len(keep_idx) + len(other_idx) == num_nodes_i

    # Calculate how many additional nodes we can include
    remaining_space = min(max_seq_len, num_nodes_i) - len(keep_idx)

    # Combine must-keep indices with as many other indices as will fit
    if remaining_space > 0:
        other_idx_selected = random.sample(other_idx, remaining_space)
        assert len(other_idx_selected) == remaining_space
        idx_nodes = keep_idx + other_idx_selected
        assert max(idx_nodes) < num_nodes_i
    else:
        idx_nodes = random.sample(keep_idx, max_seq_len)

    return sorted(idx_nodes)


# adjusted from: https://github.com/ucbrise/graphtrans/blob/main/modules/utils.py#L5
def pad_batch(
    x: torch.Tensor,
    batch_idx: torch.Tensor,
    max_seq_len: int,
    get_mask: bool = False,
    keep_indices: torch.Tensor = None,
):
    """
    Pads the batch to a fixed length because transfmrer can not handle variable-length sequences.
    For example if:
    batch = [
        [node1, node2, node3],           # 3 nodes
        [node1, node2, node3, node4, node5]  # 5 nodes
    ]
    needs to be padded to:
    batch = [
        [node1, node2, node3, 0, 0],  # 3 nodes
        [node1, node2, node3, node4, node5]  # 5 nodes
    ]

    Input:
        x: torch.Tensor of shape [N, E]
            gene-expression values X
        batch_idx: torch.Tensor of shape [N]
            batch index for each "cell" token
        max_seq_len: int
            maximum sequence length
        get_mask: bool
            whether to return the mask
        keep_indices: torch.Tensor
            indices of nodes to keep [S] (batch.mask if masked_nodes is True) if get_mask is True
    Output:
        padded_x: torch.Tensor of shape [S, B, E]
            representation values for each node such that sequence is not larger than max_input_len
        src_padding_mask: torch.Tensor of shape [num_batch, max_input_len]
            Indicates padded masked (1 = True) and unpadded (0 = False) nodes.
        index_nodes: List[List[int]]
            indices of nodes to keep [B, S] if get_mask is True
        num_nodes: List[int]
            number of nodes in each batch
        masks: List[torch.Tensor]
            mask for each batch
        max_num_nodes: int
            maximum number of nodes in any batch
    """
    assert max_seq_len > 0, "max_seq_len must be greater than 0"

    num_batch = int(batch_idx[-1].item() + 1)
    num_nodes = []
    masks = []
    index_nodes = []

    for i in range(num_batch):
        mask = batch_idx.eq(i)  # torch.tensor(Bool) [B]
        num_nodes_i = mask.sum().item()
        num_nodes.append(num_nodes_i)
        masks.append(mask)

    # get max number of nodes in any batch or maximum sequence length
    # transformer needs equal-sized sequences as input but agnostic across batches
    max_num_nodes = min(max(num_nodes), max_seq_len)

    # initialize padded_h_node with 0.0 and src_padding_mask with False (valid node)
    padded_x = x.data.new(max_num_nodes, num_batch, x.size(-1)).fill_(0)
    src_padding_mask = x.data.new(num_batch, max_num_nodes).fill_(0).bool()

    index_nodes = []
    for i, mask in enumerate(masks):
        num_nodes_i = num_nodes[i]
        # Case 1: Number of nodes in graph exceeds maximum sequence length
        if num_nodes_i > max_num_nodes:
            # Case 1.1: Node masking of input data then select masked nodes to keep
            if get_mask:
                idx_nodes = _select_masked_nodes(keep_indices, mask, num_nodes_i, max_seq_len)
            # Case 1.2: No node masking of input data then select random nodes to keep
            else:  # no masking
                idx_nodes = random.sample(range(0, num_nodes_i), max_seq_len)

            idx_nodes.sort()
            num_nodes_i = max_num_nodes
            padded_x[-num_nodes_i:, i] = x[mask][idx_nodes]

        else:  # number nodes in graph does not exceed maximum sequence length
            idx_nodes = list(range(0, num_nodes_i))
            padded_x[-num_nodes_i:, i] = x[mask][-num_nodes_i:]

        src_padding_mask[i, : max_num_nodes - num_nodes_i] = True  # [b, s]
        index_nodes.append(idx_nodes)

    if get_mask:
        return padded_x, src_padding_mask, index_nodes, num_nodes, masks, max_num_nodes
    return padded_x, src_padding_mask, index_nodes, num_nodes, None, max_num_nodes
