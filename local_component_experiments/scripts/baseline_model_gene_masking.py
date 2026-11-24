"""Using an idea from NCEM
Specifically focusing on the regression part, adapted to gene masking

Gene masking hides single (cell, gene) entries, not whole cells,
so a cell keeps most of its genes, and they become an input too
Output = a guess for each hidden gene of a cell
Every gene g has its own formula, used only when g is hidden:
y = b0 + a1*own1 + ... + c1*hid1 + ... + d1*nb1 + ... + dp*nbp + e = gene g of one cell, where g is hidden
own = the cell's other genes, as the GNN sees them (-1 where hidden)
hid = 1 if that other gene is hidden too, 0 if visible
nb = the neighbours' average of each gene, only over neighbours where that gene is visible
p = the number of genes
b0 = the offset = the general lvl of gene g
a, c, d = the weights of how much each input counts for gene g
e = error = what the formula cannot predict

Each gene's model is trained only on cells where that gene is hidden,
otherwise it would learn to copy the answer from the input
So there are p models (one per gene), and all cells share them
"""

import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import anndata as ad
import numpy as np
import squidpy as sq
import torch
from lightning.pytorch.trainer import seed_everything
from sklearn.linear_model import Ridge
from torch_geometric.utils import remove_self_loops

from interscale.config import load_config
from interscale.geome_dataloader import GraphAnnDataModule
from interscale.tl import prepare_geome_dataset, remove_zero_expression_cells, set_full_reproducibility
from interscale.tl.masking import MASK_VALUE
from interscale.train._trainingplans import masked_regression_metrics

# how many times REDRAWing is happening
# every gene model gets about as many rows as the node script had,
# but each redraw now stores every train cell (about 8x the memory), so lower this if needed
REDRAW = 10
# different penalty strength to try for large weights
# adding cost to big weights
ALPHAS = [0.1, 1, 10, 100]
# without and with the neighbours
VERSIONS = ["cell alone", "cell + neighbours"]

# 1. Training set up
# gene masking only
cfg = load_config(sys.argv[1])
# checking the yaml file
if cfg.dataset.mask_strategy != "gene":
    raise SystemExit("This baseline is for gene masking")
# same order as main.py: seed first, then segmentation noise if the config asks for it,
# so on noise runs the baseline gets the same noisy data as the GNN
set_full_reproducibility(cfg.optim.seed)
adata = remove_zero_expression_cells(ad.read_h5ad(cfg.dataset.h5ad_data))
if cfg.dataset.segmentation_robustness is not None:
    # imported only here: on public main, interscale.pp imports files that are not committed,
    # so importing it at the top would break the runs without noise too
    from interscale.pp import apply_segmentation_noise

    node_fraction, overflow_fraction = cfg.dataset.segmentation_robustness
    if cfg.dataset.connectivity_key is None:
        # no stored graph: build one, like main.py
        sq.gr.spatial_neighbors(adata, **cfg.dataset.spatial_neigbors_kwargs)
        adjacency_key = "spatial_connectivities"
    else:
        # stored graph: add the noise along it, like main.py
        adjacency_key = cfg.dataset.connectivity_key
    adata = apply_segmentation_noise(adata, node_fraction, overflow_fraction, adjacency_key=adjacency_key)
datas, _ = prepare_geome_dataset(adata, cfg)

dm = GraphAnnDataModule(
    datas=datas,
    num_workers=0,
    batch_size=int(cfg.dataset.batch_size),
    mask_percentage=cfg.dataset.mask_percentage,
    mask_strategy=cfg.dataset.mask_strategy,
    learning_type="node",
)
# seeding again right before the masks are drawn, like model.train() does,
# so the val masks are the ones the GNN is scored on
seed_everything(cfg.optim.seed, workers=True)
# building the train and validation loaders and drawing masks
dm.setup("fit")
# average train cell, one number per gene
train_mean = torch.cat([d.x for d in dm.train_data]).mean(0)
n_genes = train_mean.shape[0]


def neigh_avg(x, edge_index, gene_mask):
    # averaging each gene separately, only over the neighbours where that gene is visible
    # no self loops: the cell's own genes are already an input (Nicheformer also leaves them out)
    source, target = remove_self_loops(edge_index)[0]
    # 1 where visible, 0 where hidden
    visible = (~gene_mask).to(x.dtype)
    # hidden entries are multiplied by 0, so they add nothing to the sum
    sums = torch.zeros_like(x).index_add_(0, target, (x * visible)[source])
    # counting, per cell and per gene, how many neighbours have that gene visible
    n = torch.zeros_like(x).index_add_(0, target, visible[source])
    avg = sums / n.clamp(min=1)
    # falling back to that gene's train mean when no neighbour has it visible
    return torch.where(n == 0, train_mean, avg)


# rows used for training, one per (redraw, train cell)
# every cell is kept, not only the hidden ones like in the node script,
# because under gene masking every cell has some hidden genes
own_parts = []
hid_parts = []
nb_parts = []
y_parts = []

# training
# redrawing the masks 10 times
for _ in range(REDRAW):
    # hiding a new random 20% of entries in every training graph
    dm.resample_train_mask()
    for d in dm.train_data:
        # gene_mask, not d.mask: under gene masking d.mask is True for every cell
        gene_mask = d.gene_mask.bool()
        # the cell's own genes as the GNN sees them, -1 where hidden (same as interscale's apply_mask)
        own_parts.append(torch.where(gene_mask, MASK_VALUE, d.x))
        hid_parts.append(gene_mask)
        nb_parts.append(neigh_avg(d.x, d.edge_index, gene_mask))
        # real expression
        y_parts.append(d.x)
OWN = torch.cat(own_parts).numpy()
HID = torch.cat(hid_parts).numpy()
NB = torch.cat(nb_parts).numpy()
Y = torch.cat(y_parts).numpy()
rows_per_gene = HID.sum(0)
print(f"training rows: {Y.shape[0]} (redraw, cell) pairs, {n_genes} genes")
print(f"rows per gene model (where that gene is hidden): {rows_per_gene.min()} to {rows_per_gene.max()}")

# validation
# the val masks were drawn once in dm.setup("fit") and stay the same
batch_sizes, own_parts, hid_parts, nb_parts, y_parts = [], [], [], [], []
for batch in dm.val_dataloader():
    gene_mask = batch.gene_mask.bool()
    own_parts.append(torch.where(gene_mask, MASK_VALUE, batch.x))
    hid_parts.append(gene_mask)
    nb_parts.append(neigh_avg(batch.x, batch.edge_index, gene_mask))
    y_parts.append(batch.x)
    batch_sizes.append(batch.num_nodes)
OWN_VAL = torch.cat(own_parts).numpy()
HID_VAL = torch.cat(hid_parts).numpy()
NB_VAL = torch.cat(nb_parts).numpy()
Y_VAL = torch.cat(y_parts).numpy()
print(f"val batches: {len(batch_sizes)}")
print(f"val: {Y_VAL.shape[0]} cells, {HID_VAL.sum()} hidden entries to predict")


def design(own, hid, nb, rows, g, version):
    # The input columns of gene g's model, for the chosen rows.
    # leaving out gene g's own columns: in these rows g is always hidden (-1 and 1), so they tell nothing
    others = np.arange(n_genes) != g
    parts = [own[rows][:, others], hid[rows][:, others]]
    if version == "cell + neighbours":
        parts.append(nb[rows])
    return np.hstack(parts)


def fit_and_predict(version, alpha):
    # Fits one model per gene and puts its guesses at the hidden val entries.
    # visible entries stay 0, the masked metrics ignore them
    pred = np.zeros_like(Y_VAL)
    for g in range(n_genes):
        # training rows for gene g: every (redraw, cell) where g is hidden
        rows = HID[:, g]
        # val rows for gene g: every val cell where g is hidden
        rows_val = HID_VAL[:, g]
        if not rows_val.any():
            continue
        # one formula for gene g. It chooses b0 and one weight per input column
        ridge = Ridge(alpha=alpha).fit(design(OWN, HID, NB, rows, g, version), Y[rows, g])
        pred[rows_val, g] = ridge.predict(design(OWN_VAL, HID_VAL, NB_VAL, rows_val, g, version))
    return pred


def score(pred):
    """Masked metrics per val batch, then the plain mean over batches, like in logs"""
    pearsons, r2s = [], []
    start = 0
    for n in batch_sizes:
        rows = slice(start, start + n)
        m = masked_regression_metrics(
            torch.from_numpy(pred[rows]).double(),
            torch.from_numpy(Y_VAL[rows]).double(),
            torch.from_numpy(HID_VAL[rows]),
        )
        pearsons.append(m["pearson_corr"].item())
        r2s.append(m["r2"].item())
        start += n
    return np.mean(pearsons), np.mean(r2s)


# fit and score
print(f"\n{'':28s}{'val_masked_pearson_corr':>25s}{'val_masked_r2':>15s}")
# sanity check, not a baseline: the cell + neighbours model contains this column, so it usually
# scores at least as well. If it scores clearly lower, the train and val cells probably differ
# the plain average adapts to each place, the fitted weights don't
p, r = score(NB_VAL)
print(f"{'plain neighbour average':28s}{p:25.3f}{r:15.3f}")
for version in VERSIONS:
    for alpha in ALPHAS:
        p, r = score(fit_and_predict(version, alpha))
        print(f"{f'{version}, a={alpha}':28s}{p:25.3f}{r:15.3f}")
