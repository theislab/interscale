import random
import torch
import torchmetrics
from scipy.stats import pearsonr

def define_loss(cfg, class_weights):
    if 'classification' in cfg.dataset.prediction_task:
        if cfg.optim.loss == 'CrossEntropy':
            return torch.nn.CrossEntropyLoss()
        elif cfg.optim.loss == 'WeightedCE':
            return torch.nn.CrossEntropyLoss(torch.from_numpy(class_weights))
        else:
            raise Exception("Classification must be run with CrossEntropy or WeightedCE loss.")
    elif 'regression' in cfg.dataset.prediction_task:
        if cfg.optim.loss == 'MSELoss':
            return torch.nn.MSELoss()
        elif cfg.optim.loss == 'GaussianNLL':
            return torch.nn.GaussianNLLLoss()
        elif cfg.optim.loss == 'SmoothL1':
            return torch.nn.SmoothL1Loss()
        else:
            raise Exception("Regression must be run with MSELoss, GaussianNLL or SmoothL1 loss.")
    else:
        raise Exception("Prediction task must define 'classification' or 'regression'.")
    
def define_classification_metrics(cfg):
    accurary = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.dataset.num_classes)
    f1_score_micro = torchmetrics.F1Score(task="multiclass", num_classes=cfg.dataset.num_classes, average="micro")
    f1_score_macro = torchmetrics.F1Score(task="multiclass", num_classes=cfg.dataset.num_classes, average="macro")
    f1_score_per_class = torchmetrics.F1Score(task="multiclass", num_classes=cfg.dataset.num_classes, average=None)
    return accurary, f1_score_micro, f1_score_macro, f1_score_per_class

def define_regression_metrics(num_outputs):
    mse = torchmetrics.MeanSquaredError()
    r2_raw = torchmetrics.R2Score(num_outputs=num_outputs, multioutput = 'raw_values')
    r2 = torchmetrics.R2Score(num_outputs=num_outputs, multioutput = 'uniform_average')
    r2_single = torchmetrics.R2Score()
    return mse, r2_raw, r2, r2_single

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

def pad_batch(h_node, batch, max_seq_len, get_mask=False, keep_indices=None):
    """
    adjusted from: https://github.com/ucbrise/graphtrans/blob/main/modules/utils.py#L5
    Input: 
        h_node: [S, D_transformer_in]
        batch: batch index for each sequence token [S]
        keep_indices: indices of nodes to keep [S] (batch.mask if masked_nodes is True)
    Output:
        padded_h_node: representation values for each node such that sequence is not larger than max_input_len
            [max_input_len, D_transformer_in]
        src_padding_mask: indicates the transformer which nodes it should take into account for learning and which not. False = valid node, True = padded node
            [B, max_input_len]
        
        
    """
    num_batch = batch[-1].item() + 1
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
                idx_nodes = list(range(0, num_nodes_i))
                padded_h_node[-num_nodes_i:, i] = h_node[mask][-num_nodes_i:]
            
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


def str_to_int_or_none(s):
    if s is None:
        return None
    try:
        return int(s)
    except ValueError:
        return None