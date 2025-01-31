
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
    f1_score_per_class = torchmetrics.F1Score(task="multticlass", num_classes=cfg.dataset.num_classes, average=None)
    return accurary, f1_score_micro, f1_score_macro, f1_score_per_class

def define_regression_metrics(num_outputs):
    mse = torchmetrics.MeanSquaredError()
    r2_raw = torchmetrics.R2Score(num_outputs=num_outputs, multioutput = 'raw_values')
    r2 = torchmetrics.R2Score(num_outputs=num_outputs, multioutput = 'uniform_average')
    r2_single = torchmetrics.R2Score()
    pearson_corr = torchmetrics.PearsonCorrCoef(num_outputs=num_outputs)
    spearman = torchmetrics.SpearmanCorrCoef(num_outputs=num_outputs)
    return mse, r2_raw, r2, r2_single, pearson_corr, spearman

def pad_batch(h_node, batch, max_input_len, get_mask=False):
    """
    adjusted from: https://github.com/ucbrise/graphtrans/blob/main/modules/utils.py#L5
    Input: 
        h_node: [S, D_transformer_in]
        batch: graph index for each sequence token [S]
    Output:
        padded_h_node: representation values for each node such that sequence is not larger than max_input_len
            [max_input_len, D_transformer_in]
        src_padding_mask: indicates the transformer which nodes it should take into account for learning and which not. False = valid node, True = padded node
            [B, max_input_len]
        
        
    """

    num_batch = batch[-1] + 1 
    num_nodes = []
    masks = []

    for i in range(num_batch):
        mask = batch.eq(i)
        masks.append(mask)
        num_node = mask.sum()
        num_nodes.append(num_node)

    # logger.info(max(num_nodes))
    if max_input_len:
        max_num_nodes = min(max(num_nodes), max_input_len)
    else:
        max_num_nodes = max(num_nodes)
    
    # initialize padded_h_node with 0.0 and src_padding_mask with False (valid node)
    padded_h_node = h_node.data.new(max_num_nodes, num_batch, h_node.size(-1)).fill_(0)
    src_padding_mask = h_node.data.new(num_batch, max_num_nodes).fill_(0).bool()

    index_nodes = []
    for i, mask in enumerate(masks):
        num_node = num_nodes[i]
        if num_node > max_num_nodes:
            idx_nodes = random.sample(list(range(num_node)), max_num_nodes)
            idx_nodes.sort()
            num_node = max_num_nodes 
            padded_h_node[-num_node:, i] = h_node[mask][idx_nodes]
        else:
            #idx_nodes = list(range(max_num_nodes-num_node, max_num_nodes))
            idx_nodes = list(range(0, num_node))
            padded_h_node[-num_node:, i] = h_node[mask][-num_node:]
        #padded_h_node[-num_node:, i] = h_node[mask][-num_node:]
        # src_padding_mask[i, : max_num_nodes - num_node] = True  # [b, s]
        src_padding_mask[i, : max_num_nodes - num_node] = True  # [b, s]
        index_nodes.append(idx_nodes)

    if get_mask:
        return padded_h_node, src_padding_mask, index_nodes, num_nodes, masks, max_num_nodes
    return padded_h_node, src_padding_mask, index_nodes


def str_to_int_or_none(s):
    if s is None:
        return None
    try:
        return int(s)
    except ValueError:
        return None