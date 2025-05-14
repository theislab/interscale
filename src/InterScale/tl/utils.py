import random
import torch
from scipy.stats import pearsonr

def check_and_update_cfg(cfg, 
                         prediction_task: str = None,
                         prediction_level: str = None, 
                         prediction_obs: str = None,
                         layer_key: str = None,
                        sample_key: str = None,
                        group_label: str = None):
    """Checks for jupyer notebook specifications and updates cfg if necessary."""
    
    cfg.set_new_allowed(True)
    cfg.defrost()
    if prediction_task != cfg.dataset.prediction_task:
        print(f"Update prediction task (from '{cfg.dataset.prediction_task}' to '{prediction_task}')")
        cfg.dataset.prediction_task = prediction_task
    if prediction_level != cfg.dataset.prediction_level:
        print(f"Update prediction level (from '{cfg.dataset.prediction_level}' to '{prediction_level}')")
        cfg.dataset.prediction_level = prediction_level
    if prediction_obs != cfg.dataset.prediction_obs:
        print(f"Update prediction obs (from '{cfg.dataset.prediction_obs}' to '{prediction_obs}')")
        cfg.dataset.prediction_obs = prediction_obs
    if layer_key != cfg.dataset.layer_key:
        print(f"Update layer key (from '{cfg.dataset.layer_key}' to '{layer_key}')")
        cfg.dataset.layer_key = layer_key
    if sample_key != cfg.dataset.sample_key[0]:
        print(f"Update sample key (from '{cfg.dataset.sample_key}' to '{sample_key}')")
        cfg.dataset.sample_key = sample_key
    if group_label != cfg.dataset.group_label:
        print(f"Update group label (from '{cfg.dataset.group_label}' to '{group_label}')")
        cfg.dataset.group_label = group_label
    cfg.freeze()
    return cfg

def compute_dynamic_variance(y_true, y_pred, axis=1, epsilon=1e-6):
    """
    Computes a dynamic variance estimate using both true and predicted values.
    Handles single-sample batches by using a default variance.

    Args:
        y_true (torch.Tensor): Ground truth values.
        y_pred (torch.Tensor): Model predictions.
        axis (int): Axis along which to compute variance.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        torch.Tensor: Combined variance estimate.
    """
    batch_size = y_true.size(0)
    
    if batch_size == 1:
        # For single samples, compute squared difference between pred and true
        # as a simple variance estimate
        diff = (y_true - y_pred) ** 2
        default_var = diff.mean(dim=axis, keepdim=True) + epsilon
        return default_var
    
    # Normal case with multiple samples
    var_true = torch.var(y_true, dim=axis, unbiased=False, keepdim=True)
    var_pred = torch.var(y_pred, dim=axis, unbiased=False, keepdim=True)
    combined_var = 0.5 * (var_true + var_pred) + epsilon

    return combined_var.squeeze()

def create_transformer_attention_mask_from_edges(edge_index: torch.Tensor, num_nodes: int, batch: torch.Tensor, index_nodes: list, num_heads: int) -> torch.Tensor:
    """
    Creates an attention mask that is inverse to the edge indices.
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
    num_batch = int(batch[-1].item() + 1)
    max_seq_len = max(len(nodes) for nodes in index_nodes)
    
    # Initialize with 1s (no attention allowed) 
    attention_mask = torch.zeros((num_batch*num_heads, max_seq_len+1, max_seq_len+1), device=edge_index.device)
    
    # Create full adjacency matrix + 1 for cls token (end of sequence)
    adj_matrix = torch.ones((num_nodes, num_nodes), device=edge_index.device) 
    adj_matrix[edge_index[0], edge_index[1]] = 1  
    
    # For each batch, extract the submatrix for kept nodes
    for b in range(num_batch):
        nodes = index_nodes[b]
        seq_len = len(nodes)
        # Extract submatrix for the kept nodes
        batch_mask = adj_matrix[nodes][:, nodes]  # Get submatrix for kept nodes
        # Add row and column of ones for CLS token - full attention
        batch_mask = torch.cat([batch_mask, torch.zeros(batch_mask.size(0), 1, device=batch_mask.device)], dim=1)  # Add column
        batch_mask = torch.cat([batch_mask, torch.zeros(1, batch_mask.size(1), device=batch_mask.device)], dim=0)  # Add row
        attention_mask[b*num_heads:b*num_heads+num_heads, :seq_len+1, :seq_len+1] = batch_mask
    return attention_mask

def pad_batch(h_node, batch, max_seq_len: int, get_mask=False, keep_indices=None):
    """
    adjusted from: https://github.com/ucbrise/graphtrans/blob/main/modules/utils.py#L5
    Input: 
        h_node: [S, D_transformer_in]
        batch: batch index for each sequence token [S]
        max_seq_len: maximum sequence length 
        keep_indices: indices of nodes to keep [S] (batch.mask if masked_nodes is True)
    Output:
        padded_h_node: representation values for each node such that sequence is not larger than max_input_len
            [max_input_len, D_transformer_in]
        src_padding_mask: indicates the transformer which nodes it should take into account for learning and which not. False = valid node, True = padded node
            [B, max_input_len]
        index_nodes: indices of nodes to keep [B, S]
        
    """
    num_batch = int(batch[-1].item() + 1)
    num_nodes = []
    masks = []
    index_nodes = []

    for i in range(num_batch):
        mask = batch.eq(i) # torch.tensor(Bool) [B]
        num_nodes_i = mask.sum().item() 
        num_nodes.append(num_nodes_i)
        masks.append(mask)

    # logger.info(max(num_nodes))
    if max_seq_len:
        max_num_nodes = min(max(num_nodes), max_seq_len)
    else:
        max_num_nodes = max(num_nodes)
    
    # initialize padded_h_node with 0.0 and src_padding_mask with False (valid node)
    padded_h_node = h_node.data.new(max_num_nodes, num_batch, h_node.size(-1)).fill_(0)
    src_padding_mask = h_node.data.new(num_batch, max_num_nodes).fill_(0).bool()

    index_nodes = []
    for i, mask in enumerate(masks):
        num_nodes_i = num_nodes[i]
        if num_nodes_i > max_num_nodes:
            if get_mask:
                must_keep = keep_indices[mask] # torch.tensor(Bool) [G]
                other_nodes = ~must_keep # torch.tensor(Bool)
                
                # Get indices of must-keep nodes and other nodes separately
                # keep_idx = torch.where(mask)[0][must_keep].tolist()
                # other_idx = torch.where(mask)[0][other_nodes].tolist()
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
                    
            else: # no masking
                idx_nodes = random.sample(range(0, num_nodes_i), max_seq_len)
            
            idx_nodes.sort()
            num_nodes_i = max_num_nodes 
            padded_h_node[-num_nodes_i:, i] = h_node[mask][idx_nodes]
        
        else: # number nodes in graph does not exceed maximum sequence length
            idx_nodes = list(range(0, num_nodes_i))
            padded_h_node[-num_nodes_i:, i] = h_node[mask][-num_nodes_i:]

        src_padding_mask[i, : max_num_nodes - num_nodes_i] = True  # [b, s]
        index_nodes.append(idx_nodes)

    if get_mask:
        return padded_h_node, src_padding_mask, index_nodes, num_nodes, masks, max_num_nodes
    return padded_h_node, src_padding_mask, index_nodes, num_nodes, None, max_num_nodes
