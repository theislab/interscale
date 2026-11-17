"""Using an idea from NCEM
Specifically focusing on the regression part
Input = average expression of its visible neighbors
Output = a guess for each of the cell's genes
The guess for each gene is a weighted sum of all inputs plus an offset
y = b0 + b1*x1 + b2*x2 + ... + bp*xp + e = one gene of one hidden cell
x1 ... xp = visible neighbors' average
p = the number of genes
b0 = the offset = the general lvl of gene A
b1 ... bp = the weights of how much each neighbour gene counts for this specific gene
e = error = what the formula cannot predict

Here we will have node masking only
"""

import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import anndata as ad
import numpy as np
import torch
from lightning.pytorch.trainer import seed_everything
from sklearn.linear_model import Ridge

from interscale.config import load_config
from interscale.geome_dataloader import GraphAnnDataModule
from interscale.tl import prepare_geome_dataset, remove_zero_expression_cells
from interscale.train._trainingplans import masked_regression_metrics

# how many times REDRAWing is happening
REDRAW = 10
# different penalty strength to try for large weights
# adding cost to big weights
ALPHAS = [0.1, 1, 10, 100]

# 1. Training set up
# node masking only
cfg = load_config(sys.argv[1])
# checking the yaml file
if cfg.dataset.mask_strategy != "node":
    raise SystemExit("This baseline is for node masking")
# to verify
adata = remove_zero_expression_cells(ad.read_h5ad(cfg.dataset.h5ad_data))
datas, _ = prepare_geome_dataset(adata, cfg)

dm = GraphAnnDataModule(
    datas=datas,
    num_workers=0,
    batch_size=int(cfg.dataset.batch_size),
    mask_percentage=cfg.dataset.mask_percentage,
    mask_strategy=cfg.dataset.mask_strategy,
    learning_type="node",
)
# default to 40 for reproducibility
seed_everything(cfg.optim.seed, workers=True)
# building the train and validation loaders and drawing masks
dm.setup("fit")
# average train cell
train_mean = torch.cat([d.x for d in dm.train_data]).mean(0)


def neigh_avg(x, edge_index, hidden):
    # different cells can have different number of neighbors
    # but we need 88 inputs, so we just average all
    # splitting edges into senders and receivers
    source, target = edge_index
    # only visible neighbors are included
    keep = ~hidden[source]
    # starts from 0, for every kept edge, adds the sender's expression to the receiver's row
    sums = torch.zeros_like(x).index_add_(0, target[keep], x[source[keep]])
    n = torch.zeros(x.shape[0]).index_add_(0, target[keep], torch.ones(int(keep.sum())))
    avg = sums / n.clamp(min=1).unsqueeze(1)
    # falling back to the average train cell here when no visible neighbors
    avg[n == 0] = train_mean
    return avg


# rows used for training
x_parts = []
y_parts = []

# training
# shuffle 10 times
for _ in range(REDRAW):
    # hiding new random 20% of cells in every training graph
    dm.resample_train_mask()
    for d in dm.train_data:
        hidden = d.mask.bool()
        # input
        x_parts.append(neigh_avg(d.x, d.edge_index, hidden)[hidden])
        # real expression
        y_parts.append(d.x[hidden])
X = torch.cat(x_parts).numpy()
Y = torch.cat(y_parts).numpy()
print(f"training rows: {X.shape[0]} hidden cells, {X.shape[1]} genes")

# validation
val_batches = []
for batch in dm.val_dataloader():
    hidden = batch.mask.bool()
    # storing neighb averages (x), the real expression (y), and which cells are hidden
    val_batches.append((neigh_avg(batch.x, batch.edge_index, hidden), batch.x, hidden))


# scoring metrics
def score(predict):
    """Mean over the val batches of the masked metrics."""
    pearson = []
    r2 = []
    # going through stored val batches
    for x_nb, x_true, hidden in val_batches:
        pred = predict(x_nb)
        m = masked_regression_metrics(pred, x_true, hidden.unsqueeze(1).expand_as(x_true))
        pearson.append(m["pearson_corr"].item())
        r2.append(m["r2"].item())
    return np.mean(pearson), np.mean(r2)

# fit and score
print(f"\n{'':26s}{'val_masked_pearson_corr':>25s}{'val_masked_r2':>15s}")
p, r = score(lambda x_nb: x_nb)
print(f"{'plain neighbour average':26s}{p:25.3f}{r:15.3f}")
for alpha in ALPHAS:
    # fitting all 88 equations at once. It chooses b0 (intercept) and b1...b88 for every gene
    ridge = Ridge(alpha=alpha).fit(X, Y)
    p, r = score(lambda x_nb, model=ridge: torch.tensor(model.predict(x_nb.numpy()), dtype=torch.float32))
    print(f"{f'linear regression, a={alpha}':26s}{p:25.3f}{r:15.3f}")
