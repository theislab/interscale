"""Comparing each checkpoint's decoder bias with its starting values

Usage examples:
    python check_ckpt_bias.py configs/legnini_gat_delaunay.yaml results/checkpoints_e64/
    python check_ckpt_bias.py configs/legnini_gat_delaunay.yaml results/checkpoints_e64/legnini_delaunay_*_GAT_*.ckpt
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from interscale.config import load_config
from interscale.tl import remove_zero_expression_cells

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("config", help="YAML config the runs were trained with")
parser.add_argument("checkpoints", nargs="+", help="checkpoint files, or folders that contain them")
args = parser.parse_args()

# Data settings come from the same config as training, so the starting values match.
cfg = load_config(args.config)
adata = remove_zero_expression_cells(ad.read_h5ad(cfg.dataset.h5ad_data))  # main_sweep.py does this before training
X = adata.layers[cfg.dataset.layer_key] if cfg.dataset.layer_key else adata.X
is_train = (adata.obs[cfg.dataset.split_key] == "train").to_numpy()
bias_start = np.asarray(X[is_train].mean(axis=0)).ravel()

paths = []
for p in map(Path, args.checkpoints):
    paths += sorted(p.glob("*.ckpt")) if p.is_dir() else [p]

for path in paths:
    if path.name.endswith("model.ckpt"):
        continue
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    bias = ckpt["state_dict"].get("module.decoder.decoder.bias")
    if bias is None or bias.numel() != bias_start.size:  # not a linear decoder, or a different dataset/task
        continue
    change = np.abs(bias.numpy() - bias_start)
    print(f"{path.name}: epoch {ckpt['epoch']}, mean change {change.mean():.4f}, max change {change.max():.4f}")
