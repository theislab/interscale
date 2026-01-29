from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def log(t, eps=1e-20):
    """Custom log function clamped to minimum epsilon."""
    return torch.log(t.clamp(min=eps))


def poisson_loss(pred: torch.Tensor, target: torch.Tensor):
    """Poisson loss"""
    return (pred - target * log(pred)).mean()


def nonzero_median(tensor: torch.Tensor, axis: int, keepdim: bool) -> torch.Tensor:
    """Compute the median across non-zero float elements.

    Notes
    -----
    Modifies the tensor in place to avoid making a copy.
    """
    tensor = torch.where(tensor != 0.0, tensor.double(), float("nan"))

    # returns values and indices - we only want the value(s)
    medians = torch.nanmedian(tensor, dim=axis, keepdim=keepdim)[0]

    medians = medians.nan_to_num(0)

    return medians


class BalancedPearsonCorrelationLoss(torch.nn.Module):
# adjusted from seq2cells
# https://github.com/GSK-AI/seq2cells/blob/main/seq2cells/metrics_and_losses/losses.py
# accessed on 28 August 2025
    """Pearson Corr balances between across gene and cell performance"""

    def __init__(
        self,
        cross_corr: Literal["gene", "cell"] | None = None,
        norm_by: Literal["mean", "nonzero_median"] = "mean",
        eps: float = 1e-8,
    ):
        """Initialise PearsonCorrelationLoss.

        Parameter
        ---------
        rel_weight_gene: float = 1.0
            The relative weight to put on the across gene/tss correlation.
        rel_weight_cell: float = 1.0
            The relative weight to put on the across cells correlation.
        norm_by:  Literal['mean', 'nonzero_median'] = 'nonzero_median'
            What to use as across gene / cell average to subtract from the
            signal to normalise it. Mean or the Median of the non zero entries.
        eps: float 1e-8
            epsilon
        """
        super().__init__()
        self.eps = eps
        self.norm_by = norm_by
        self.cross_corr = cross_corr
        
        if self.cross_corr == 'gene':
            self.rel_weight_gene = 1.0
            self.rel_weight_cell = 0.0
        elif self.cross_corr == 'cell':
            self.rel_weight_gene = 0.0
            self.rel_weight_cell = 1.0
        elif self.cross_corr is None:
            self.rel_weight_gene = 1.0
            self.rel_weight_cell = 1.0
        else:
            raise ValueError("cross_corr must be either 'gene' or 'cell' or None.")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward.

        Notes
        -----
        preds: torch.Tensor
            2D torch tensor [genes x cells], batched over genes.
        targets: torch.Tensor
            2D torch tensor [genes x cells], batched over genes.
        """
        if self.norm_by == "mean":
            preds_avg_gene = preds.mean(dim=0, keepdim=True)
            targets_avg_gene = targets.mean(dim=0, keepdim=True)
            preds_avg_cell = preds.mean(dim=1, keepdim=True)
            targets_avg_cell = targets.mean(dim=1, keepdim=True)
        else:
            preds_avg_gene = nonzero_median(preds, 0, keepdim=True)
            targets_avg_gene = nonzero_median(targets, 0, keepdim=True)
            preds_avg_cell = nonzero_median(preds, 1, keepdim=True)
            targets_avg_cell = nonzero_median(targets, 1, keepdim=True)

        r_tss = torch.nn.functional.cosine_similarity(
            preds - preds_avg_gene,
            targets - targets_avg_gene,
            eps=self.eps,
            dim=0,
        )

        r_celltype = torch.nn.functional.cosine_similarity(
            preds - preds_avg_cell,
            targets - targets_avg_cell,
            eps=self.eps,
        )

        loss = self.rel_weight_gene * (1 - r_tss.mean()) + self.rel_weight_cell * (
            1 - r_celltype.mean()
        )

        # norm the loss to 2 by half the sum of the relative weights
        loss = (loss * 2) / (self.rel_weight_gene + self.rel_weight_cell)

        return loss
    
class SCELoss(torch.nn.Module):
    # adjusted from GraphMAE
    # https://github.com/THUDM/GraphMAE/blob/b14f080c919257b495e3cb64742884d5252d6a635/graphmae/models/loss_func.py#L5
    # accessed on 10 September 2025
    
    def __init__(self, alpha=3):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x, y):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        # loss =  - (x * y).sum(dim=-1)
        # loss = (x_h - y_h).norm(dim=1).pow(alpha)

        loss = (1 - (x * y).sum(dim=-1)).pow_(self.alpha)

        loss = loss.mean()
        return loss
class SCE_EntropyATT_Loss(torch.nn.Module):
    
    def __init__(self, alpha=3, beta=1, use_last_layer_only=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.use_last_layer_only = use_last_layer_only
        
    def forward(self, x, y, attn):

        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        loss_recon = (1 - (x * y).sum(dim=-1)).pow_(self.alpha).mean()

        if self.use_last_layer_only and attn.dim() == 5:
            attn = attn[-1] # [Batch, Heads, N, N]

        attn = torch.clamp(attn, min=1e-9, max=1.0)

        attn_entropy = torch.special.entr(attn) 
        
        loss_attn = attn_entropy.sum(dim=-1).mean()
        
        return loss_recon + self.beta * loss_attn
        #return loss_attn

class GaussianLoss(torch.nn.Module):
    # adjusted from NCEM
    # https://github.com/theislab/ncem/blob/main/ncem/utils/losses.py
    # accessed on 01 September 2025
    
    """Custom gaussian loss."""
    
    def __init__(self, cross_corr: Literal["gene", "cell"]):
        super().__init__()
        self.cross_corr = cross_corr

    def forward(self, y_true: torch.Tensor, 
                y_pred: torch.Tensor) -> torch.Tensor:
        """Implement Gaussian loss as reconstruction loss.
        Exception: If axis of calculation has sd = 0, return 0. each part of the sum is -inf and inf = 0. Otherwise the sum is NaN.

        Parameters
        ----------
        y_true : torch.Tensor [N, F]
            Ground truth values
        y_pred : torch.Tensor [N, F]
            Predicted values.

        Returns
        -------
        neg_ll : torch.Tensor [1]
            Gaussian loss as reconstruction loss
        """
        if self.cross_corr == "gene":
            axis = 1
        elif self.cross_corr == "cell":
            axis = 0
        else:   
            raise ValueError("cross_corr must be either 'gene' or 'cell'.")

        sd = torch.std(y_true, dim=axis, keepdim=True)
        
        # Normal case - no zero variance
        neg_ll = (torch.log(torch.sqrt(torch.tensor(2 * torch.pi)) * sd) + 
                    0.5 * torch.square(y_pred - y_true) / torch.square(sd))
            
        print(neg_ll)
            
        neg_ll = torch.sum(neg_ll, dim=axis)  # sum across output features
        return neg_ll.mean() 