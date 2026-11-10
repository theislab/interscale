"""Preparing Legnnin et al. 2023 data for InterScale.
Following Interscale's tutorial "Set up and download - Molecular cartography data fro Legnini et al., 2023

Need to run only once from local_component_experiments

    python scripts/prepare_legnini.py

Writes data/legnini_prepared.h5ad
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq

warnings.filterwarnings("ignore")

from interscale import datasets as ds
from interscale.pp import compute_neighborhood_stats, remove_zero_expression_cells


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
        # prings the average number of neighbors for that map
        print(f"{name}: {degree.mean():.1f} neighbours per cell on average")


# hardcoding the path for now
DATA = "legnini"
RAW = Path(f"data/raw/{DATA}.h5ad")
OUT = Path(f"data/{DATA}_prepared.h5ad")

# splitting data into training, testing, and validation
# hard coding for now, just to keep everything consistent
train_samples = [
    "slide1_D2-2",
    "slide1_D2-3",
    "slide1_C2-2",
    "slide4_B2-1",
    "slide1_A2-1",
    "slide1_C2-1",
    "slide4_A2-1",
    "slide4_A2-2",
    "slide1_C2-5",
    "slide1_B2-2",
]
test_samples = ["slide4_A2-3", "slide4_B2-2"]
validation_samples = ["slide1_B2-1", "slide1_B2-3", "slide1_A2-2", "slide1_C2-3", "slide4_B2-3"]

# load the data
adata = ds.legnini(RAW)
# check if everything is okay
print(adata)
print("Zero count cells: ", (adata.X.sum(1) == 0).sum())
print(f"NaN values to begin with: {np.isnan(adata.obsm['spatial']).sum()}")

# removing all cells tha thave entry NaN
valid_coords = ~np.isnan(adata.obsm["spatial"]).any(axis=1)
adata = adata[valid_coords].copy()
# Verify the removal of NaN values
print(f"NaN values remaining: {np.isnan(adata.obsm['spatial']).sum()}")

# removing all cells with zero count
adata = remove_zero_expression_cells(adata)

# 1. Normalization = data needs to be normalized for Interscale
# Ideally, counts should be normalized between 0 and 3

# shifted Logarithm
scales_counts = sc.pp.normalize_total(adata, target_sum=10000, inplace=False)
# log1p transform
adata.layers["log1p_norm"] = sc.pp.log1p(scales_counts["X"], copy=True)

print("Raw - Min: ", {adata.layers["raw"].min()}, ", Max: ", {adata.layers["raw"].max()})
print("Log1pNorm - Min: ", {adata.layers["log1p_norm"].min()}, ", Max: ", {adata.layers["log1p_norm"].max()})

# 2. Calculate spatial connectivity matrix (check)
compute_neighborhood_stats(adata, radii=[0, 200, 300], library_key="sample")
# Make sure that obs_names are unique and convertable to string.
adata.obs_names_make_unique()
adata.obs["obs_names"] = adata.obs_names

# 3. Sliding window (for global only (double check)) = skipped

# 4. Split data into train and val and test
adata.obs["split"] = "unassigned"  # starting every cell with unassigned
adata.obs.loc[adata.obs["sample"].isin(train_samples), "split"] = "train"
adata.obs.loc[adata.obs["sample"].isin(validation_samples), "split"] = "val"
adata.obs.loc[adata.obs["sample"].isin(test_samples), "split"] = "test"
if (adata.obs["split"] == "unassigned").any():
    raise ValueError("Some cells were not assigned to train, val or test.")
df = adata.obs[["split", "sample", "condition"]]
value_counts = pd.DataFrame(df.values, columns=df.columns).value_counts()
print(value_counts)

# 5. adding graphs
add_graphs(adata, library_key="sample", radius=200, n_neighs=10)

# saving prepped data
OUT.parent.mkdir(parents=True, exist_ok=True)
adata.write(OUT)
print(f"\n Saved {OUT}: {adata.n_obs} cells x {adata.n_vars} genes")
