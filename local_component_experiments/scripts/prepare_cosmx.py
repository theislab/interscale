"""Preparing Melton et al. 2025 CosMx data (pancreas, ND vs T1D) for InterScale.
The same steps as scripts/prepare_legnini.py; the lines that are specific to CosMx are marked "CosMx:".

Need to run only once from local_component_experiments

    python scripts/prepare_cosmx.py

Writes data/cosmx_prepared.h5ad
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

warnings.filterwarnings("ignore")

from interscale.pp import compute_neighborhood_stats
from interscale.tl import remove_zero_expression_cells


def add_graphs(adata, library_key, radius, n_neighs):
    """Build the radius, kNN and Delaunay graphs once, per library_key group, and keep them in adata.obsp."""
    adata.obs[library_key] = adata.obs[library_key].astype("category")
    common = dict(coord_type="generic", library_key=library_key)
    sq.gr.spatial_neighbors(adata, radius=radius, key_added="radius", **common)
    sq.gr.spatial_neighbors(adata, n_neighs=n_neighs, key_added="knn", **common)
    sq.gr.spatial_neighbors(adata, delaunay=True, key_added="delaunay", **common)

    for name in ["radius", "knn", "delaunay"]:
        # counting how many neighbors each cell has
        degree = adata.obsp[f"{name}_connectivities"].getnnz(axis=1)
        # prints the average number of neighbors for that map
        print(f"{name}: {degree.mean():.1f} neighbours per cell on average")


# hardcoding the path for now
DATA = "cosmx"
RAW = Path("/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/data/melton25.h5ad")
OUT = Path(f"data/{DATA}_prepared.h5ad")

# splitting data into training, testing, and validation, by FOV
# hard coding for now, just to keep everything consistent
# CosMx: one sample = one FOV, named slide + FOV, e.g. "Run5211_S1_fov12". There are 71 FOVs from 6 donors:
# each slide holds one T1D and one ND donor
# proportions as Legnini (10 / 5 / 2 of 17 samples = 59 / 29 / 12 %), so every donor and both conditions
# are in train, val and test: 42 / 21 / 8 FOVs (4 T1D and 4 ND FOVs in test)
train_samples = [
    "Run5211_S1_fov1", "Run5211_S1_fov2", "Run5211_S1_fov4", "Run5211_S1_fov6", "Run5211_S1_fov9",
    "Run5211_S1_fov10", "Run5211_S1_fov11",
    "Run5211_S1_fov12", "Run5211_S1_fov13", "Run5211_S1_fov17", "Run5211_S1_fov19", "Run5211_S1_fov20",
    "Run5211_S1_fov21", "Run5211_S1_fov22",
    "Run5211_S2_fov1", "Run5211_S2_fov2", "Run5211_S2_fov5", "Run5211_S2_fov6", "Run5211_S2_fov8",
    "Run5211_S2_fov9", "Run5211_S2_fov10", "Run5211_S2_fov11",
    "Run5211_S2_fov12", "Run5211_S2_fov13", "Run5211_S2_fov14", "Run5211_S2_fov15", "Run5211_S2_fov17",
    "Run5211_S3_fov1", "Run5211_S3_fov2", "Run5211_S3_fov4", "Run5211_S3_fov6", "Run5211_S3_fov7",
    "Run5211_S3_fov10", "Run5211_S3_fov11", "Run5211_S3_fov12", "Run5211_S3_fov13",
    "Run5211_S3_fov16", "Run5211_S3_fov17", "Run5211_S3_fov18", "Run5211_S3_fov19", "Run5211_S3_fov20",
    "Run5211_S3_fov22",
]
test_samples = [
    "Run5211_S1_fov5",
    "Run5211_S1_fov16", "Run5211_S1_fov18",
    "Run5211_S2_fov3",
    "Run5211_S2_fov19",
    "Run5211_S3_fov5", "Run5211_S3_fov14",
    "Run5211_S3_fov21",
]
validation_samples = [
    "Run5211_S1_fov3", "Run5211_S1_fov7", "Run5211_S1_fov8",
    "Run5211_S1_fov14", "Run5211_S1_fov15", "Run5211_S1_fov23", "Run5211_S1_fov24",
    "Run5211_S2_fov4", "Run5211_S2_fov7", "Run5211_S2_fov21", "Run5211_S2_fov22",
    "Run5211_S2_fov16", "Run5211_S2_fov18", "Run5211_S2_fov20",
    "Run5211_S3_fov3", "Run5211_S3_fov8", "Run5211_S3_fov9", "Run5211_S3_fov25",
    "Run5211_S3_fov15", "Run5211_S3_fov23", "Run5211_S3_fov24",
]


# load the data
adata = sc.read_h5ad(RAW)
# the values are already normalized
# print(".X whole numbers:", (adata.X != adata.X.astype(int)).sum() == 0)
# print("counts whole numbers:", (adata.layers["counts"] != adata.layers["counts"].astype(int)).sum() == 0)
# CosMx: keep the raw counts in layers["raw"], as the Legnini loader does
adata.X = adata.layers["counts"].copy()
adata.layers["raw"] = adata.layers["counts"].copy()
# check if everything is okay
print(adata)
print("Zero count cells: ", (adata.X.sum(1) == 0).sum())
print(f"NaN values to begin with: {np.isnan(adata.obsm['spatial']).sum()}")

# removing all cells that have entry NaN
valid_coords = ~np.isnan(adata.obsm["spatial"]).any(axis=1)
adata = adata[valid_coords].copy()
# Verify the removal of NaN values
print(f"NaN values remaining: {np.isnan(adata.obsm['spatial']).sum()}")

# CosMx: the negative-control and system-control probes are not genes, so they are removed
controls = adata.var_names.str.startswith(("NegPrb", "SystemControl", "FalseCode"))
print(f"Control probes removed: {controls.sum()}")
adata = adata[:, ~controls].copy()

# removing all cells with zero count
adata = remove_zero_expression_cells(adata)

# CosMx: one graph per FOV. FOV numbers repeat on every slide, so the sample name combines slide and FOV
adata.obs["sample"] = (adata.obs["slide"].astype(str) + "_fov" + adata.obs["fov"].astype(str)).astype("category")

# 1. Normalization = data needs to be normalized for Interscale
# Ideally, counts should be normalized between 0 and 3

# shifted Logarithm
scales_counts = sc.pp.normalize_total(adata, target_sum=10000, inplace=False)
# log1p transform
adata.layers["log1p_norm"] = sc.pp.log1p(scales_counts["X"], copy=True)

print("Raw - Min: ", {adata.layers["raw"].min()}, ", Max: ", {adata.layers["raw"].max()})
print("Log1pNorm - Min: ", {adata.layers["log1p_norm"].min()}, ", Max: ", {adata.layers["log1p_norm"].max()})

# 2. Calculate spatial connectivity matrix (check)
# CosMx: the coordinates are not in the same units as Legnini's, so check which radius gives about 10
# neighbours per cell (in Legnini, radius 200 gave 10.1) and use that radius later in step 5
compute_neighborhood_stats(adata, radii=[100, 200, 300], library_key="sample")
# Make sure that obs_names are unique and convertable to string.
adata.obs_names_make_unique()
adata.obs["obs_names"] = adata.obs_names

# 3. Sliding window (for global only (double check)) = skipped

# 4. Split data into train and val and test
adata.obs["split"] = "unassigned"  # starting every cell with unassigned
adata.obs.loc[adata.obs["sample"].isin(train_samples), "split"] = "train"
adata.obs.loc[adata.obs["sample"].isin(validation_samples), "split"] = "val"
adata.obs.loc[adata.obs["sample"].isin(test_samples), "split"] = "test"
df = adata.obs[["split", "sample", "condition"]]
value_counts = pd.DataFrame(df.values, columns=df.columns).value_counts()
# CosMx: printed before the check, so the first run lists every FOV with its condition and number of cells
print(value_counts)
if (adata.obs["split"] == "unassigned").any():
    raise ValueError("Some cells were not assigned to train, val or test (they are listed as 'unassigned' above).")

# 5. adding graphs
# CosMx: radius = the one from the check in step 2 that gives about 10 neighbours
add_graphs(adata, library_key="sample", radius=100, n_neighs=10)

# saving prepped data
OUT.parent.mkdir(parents=True, exist_ok=True)
adata.write(OUT)
print(f"\n Saved {OUT}: {adata.n_obs} cells x {adata.n_vars} genes")
