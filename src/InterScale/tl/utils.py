import random
import torch
from scipy.stats import pearsonr

def set_full_reproducibility(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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

def create_transformer_attention_mask_from_edges(edge_index: torch.Tensor, 
                                                 num_nodes: int, 
                                                 batch: torch.Tensor, 
                                                 index_nodes: list, 
                                                 num_heads: int) -> torch.Tensor:
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
    INVALID_MASK_VALUE = -float('inf')
    
    num_batch = int(batch[-1].item() + 1)
    max_seq_len = max(len(nodes) for nodes in index_nodes)
    
    # Initialize with -inf (no attention allowed) 
    attention_mask = torch.full((num_batch*num_heads, max_seq_len+1, max_seq_len+1), INVALID_MASK_VALUE, device=edge_index.device)
    
    # Create full adjacency matrix + 1 for cls token (end of sequence)
    adj_matrix = torch.zeros((num_nodes, num_nodes), device=edge_index.device) # TODO: check if zero or ones
    adj_matrix[edge_index[0], edge_index[1]] = INVALID_MASK_VALUE
    #TODO: also fill diagonal with INVALID_MASK_VALUE
    
    # For each batch, extract the submatrix for kept nodes
    for b in range(num_batch):
        nodes = index_nodes[b]
        seq_len = len(nodes)
        assert seq_len+1 <= max_seq_len+1, f"Mismatch: seq_len+1: {seq_len+1}, max_seq_len+1: {max_seq_len+1}"
        # Extract submatrix for the kept nodes
        batch_mask = adj_matrix[nodes][:, nodes]  # Get submatrix for kept nodes
        # INSERT_YOUR_CODE
        assert torch.any(batch_mask != 0), "batch_mask contains only zero entries"
        # Add row and column of ones for CLS token - full attention
        batch_mask = torch.cat([batch_mask, torch.zeros(batch_mask.size(0), 1, device=batch_mask.device)], dim=1)  # Add column
        batch_mask = torch.cat([batch_mask, torch.zeros(1, batch_mask.size(1), device=batch_mask.device)], dim=0)  # Add row
        assert batch_mask.shape == (seq_len+1, seq_len+1), f"Mismatch: batch_mask.shape: {batch_mask.shape}, (seq_len+1, seq_len+1): {(seq_len+1, seq_len+1)}"
        assert attention_mask.shape[-2:] == (max_seq_len+1, max_seq_len+1), f"Mismatch: attention_mask.shape[-2:]: {attention_mask.shape[-2:]}, (seq_len+1, seq_len+1): {(seq_len+1, seq_len+1)}"
        # append inverse adjacency matrix to the end of the attention mask
        attention_mask[b*num_heads:b*num_heads+num_heads, -(seq_len+1):, -(seq_len+1):] = batch_mask
    
    assert not torch.any(torch.isnan(attention_mask)), "attention_mask contains NaN values"
    print('attention_mask', attention_mask.shape, attention_mask)
    return attention_mask

def get_model_filename_prefix(cfg, local_component: bool, global_component: bool):
    """Generate the filename prefix for saving model files.
    
    Parameters
    ----------
    cfg : CN
        Configuration object containing dataset and model information
        
    Returns
    -------
    str
        Filename prefix in format: <dataset_name>_<prediction_task[:4]>_<prediction_level>_<local_component_name>_<global_component_name>_
    """
    file_name_prefix = f"{cfg.dataset.name}_{cfg.dataset.prediction_task[:4]}_{cfg.dataset.prediction_level}_{cfg.optim.seed}_"
    
    if local_component and cfg.model.local_component.name:  
        file_name_prefix = file_name_prefix + f"{cfg.model.local_component.name}_"
    if global_component and cfg.model.global_component.name:
        file_name_prefix = file_name_prefix + f"{cfg.model.global_component.name}_"
        
    return file_name_prefix

def remap_state_dict_keys(state_dict):
    """
    Remap state dictionary keys to handle naming convention changes.
    
    This function handles the transition from InterScale key names to graph_transformer_long_range_niches key names:
    - local_layers.* -> local_module.layers.*
    - global_* -> global_module.*
    
    Parameters:
    - state_dict: The state dictionary from the checkpoint
    
    Returns:
    - new_state_dict: State dictionary with remapped keys
    """
    new_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Handle local module keys (InterScale: local_layers.* -> graph_transformer: local_module.layers.*)
        if key.startswith('local_layers.'):
            new_key = key.replace('local_layers.', 'local_module.layers.')
        
        # Handle global module keys (InterScale: global_* -> graph_transformer: global_module.*)
        elif key.startswith('global_'):
            new_key = key.replace('global_', 'global_module.')
        
        new_state_dict[new_key] = value
    
    return new_state_dict

def detect_and_remap_state_dict_keys(state_dict):
    """
    Automatically detect the source of the state dict and apply appropriate remapping.
    
    This function detects whether the state dict is from InterScale or graph_transformer_long_range_niches
    and applies the appropriate key remapping.
    
    Parameters:
    - state_dict: The state dictionary from the checkpoint
    
    Returns:
    - new_state_dict: State dictionary with remapped keys
    - source_detected: String indicating the detected source ('InterScale' or 'graph_transformer')
    """
    # Check if this is an InterScale checkpoint (has local_layers or global_ keys)
    has_interscale_keys = any(key.startswith('local_layers.') or key.startswith('global_') 
                             for key in state_dict.keys())
    
    # Check if this is a graph_transformer checkpoint (has local_module or global_module keys)
    has_graph_transformer_keys = any(key.startswith('local_module.') or key.startswith('global_module.') 
                                    for key in state_dict.keys())
    
    if has_interscale_keys and not has_graph_transformer_keys:
        # This is an InterScale checkpoint, remap to graph_transformer format
        new_state_dict = remap_state_dict_keys(state_dict)
        source_detected = 'InterScale'
        print(f"Detected InterScale checkpoint format. Remapping keys to graph_transformer format.")
    elif has_graph_transformer_keys and not has_interscale_keys:
        # This is already a graph_transformer checkpoint, no remapping needed
        new_state_dict = state_dict
        source_detected = 'graph_transformer'
        print(f"Detected graph_transformer checkpoint format. No remapping needed.")
    else:
        # Mixed or unclear format, try remapping anyway
        new_state_dict = remap_state_dict_keys(state_dict)
        source_detected = 'unknown'
        print(f"Unclear checkpoint format. Attempting remapping anyway.")
    
    return new_state_dict, source_detected
