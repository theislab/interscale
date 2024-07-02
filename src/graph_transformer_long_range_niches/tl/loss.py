import torch
import torch.nn.functional as F
from collections import Counter

def weighted_cross_entropy(pred, true):
    """Weighted cross-entropy for unbalanced classes.
    """
    # calculating label weights for weighted loss computation
    print(pred)
    print(true)
    V = true.size(0)
    n_classes = pred.shape[1] if pred.ndim > 1 else 2
    label_count = torch.bincount(true)
    label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
    cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
    cluster_sizes[torch.unique(true)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes > 0).float()
    # multiclass
    if pred.ndim > 1:
        return weight
    # binary
    else:
        return None


def calculate_class_weights(y_true):
    class_counts = Counter(y_true)
    total_samples = len(y_true)

    # Calculate class weights
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    # Convert class weights to a tensor
    weights = torch.tensor([class_weights[i] for i in range(len(class_counts))], dtype=torch.float)