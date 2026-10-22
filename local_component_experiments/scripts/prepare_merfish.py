"""
Prepare the MERFISH mouse hypothalamus data (Moffitt et al. 2018) for InterScale.

Run once, from the repo root:

    python scripts/prepare_merfish.py

It downloads the data the first time (to data/raw/merfish.h5ad), cleans it, splits it by
tissue slice, and writes data/merfish_prepared.h5ad with:

    X                    log-scaled expression (never negative; InterScale marks hidden cells with -1)
    layers["original"]   the values as downloaded
    obs["Cell_class"]    cell-type labels, ambiguous cells removed
    obs["slice"]         one id per tissue slice  -> sample_key in the YAML
    obs["split"]         train / val / test, whole slices at a time
    obsm["spatial"]      2D cell coordinates within each slice

At the end it prints how many neighbours a cell gets for a few radii, so we can pick the
radius for the YAML.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from scipy import sparse
from sklearn.neighbors import KDTree

from interscale.pp import compute_neighborhood_stats

RAW = Path("data/raw/merfish.h5ad")
OUT = Path("data/merfish_prepared.h5ad")
LABEL = "Cell_class"


def look(adata):
    """Print what is in the downloaded object."""
    print(adata)  # summary
    print("\nCells per cell class:")
    print(adata.obs[LABEL].value_counts(dropna=False).to_string())  # counting missing labels too
    print("\nCells per Bregma value (one value = one slice position):")
    print(adata.obs["Bregma"].value_counts().sort_index().to_string())
    if "Animal_ID" in adata.obs:  # everything comes from the same animal
        print(f"\nAnimals: {sorted(adata.obs['Animal_ID'].astype(str).unique())}")


def drop_control_probes(adata):
    # removing Black barcodes that match no gene and are used to measure
    # false detections = noise here
    """Remove 'Blank' barcodes: negative controls, not genes."""
    blank = adata.var_names.str.lower().str.startswith("blank")  # true/false
    if blank.any():
        # keeping everything that is not blank
        adata = adata[:, ~blank].copy()
    return adata


def clean_cells(adata):
    """Keep cells with a usable label, valid coordinates and at least some expression."""
    # interscale cannot use cells with no lbale, no coordinate, or no expressions
    if "spatial" not in adata.obsm:
        adata.obsm["spatial"] = adata.obs[["Centroid_X", "Centroid_Y"]].to_numpy()
    adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]

    no_label = adata.obs[LABEL].isna() | adata.obs[LABEL].astype(str).str.lower().str.startswith("ambiguous")
    no_coords = np.isnan(adata.obsm["spatial"]).any(axis=1)
    no_expression = np.asarray(adata.X.sum(axis=1)).ravel() == 0
    # dropping if any of the three are missing
    drop = no_label.to_numpy() | no_coords | no_expression
    print(
        f"\nDropping {drop.sum()} cells: {no_label.sum()} without a usable label, "
        f"{no_coords.sum()} without coordinates, {no_expression.sum()} with no expression"
    )
    # keeping everything else
    adata = adata[~drop].copy()

    # Unused categories would still count as classes, i.e. extra outputs that can never be right.
    adata.obs[LABEL] = adata.obs[LABEL].astype("category").cat.remove_unused_categories()
    adata.uns.pop(f"{LABEL}_colors", None)
    return adata


def make_log_scaled(adata):
    # Interscale expects log-scaled expression (the Legnini tutorial uses a log1p_norm layer)
    """Bring X to log scale, keeping the downloaded values in layers['original']."""
    xmin, xmax = float(adata.X.min()), float(adata.X.max())
    if xmin < 0:
        raise SystemExit(
            "X has negative values, so it was scaled after log-transforming. InterScale needs "
            "non-negative input. Stop here and send this output to Claude."
        )
    adata.layers["original"] = adata.X.copy()

    values = adata.X.data if sparse.issparse(adata.X) else np.asarray(adata.X)
    if np.allclose(values, np.round(values)):  # are all values whole numbers?
        print("Whole numbers: raw counts. Normalising each cell to the same total, then log1p.")
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    elif xmax > 20:  # check
        print("Large values that aren't whole numbers: normalised but not logged. Applying log1p.")
        sc.pp.log1p(adata)
    else:  # in case if everything is alright
        print("Small values that aren't whole numbers: already log-scaled. Leaving X as is.")

    adata.X = adata.X.astype(np.float32)
    print(f"X now: min {float(adata.X.min()):.3g}, max {float(adata.X.max()):.3g}")
    return adata


def coordinates_in_micrometres(adata):
    """Convert coordinates to micrometres if they come in millimetres.

    Two reasons: radii become readable (29 instead of 0.029), and InterScale's config only
    accepts whole-number radii, because its default radius is the int 50 and yacs refuses a
    float in its place (tried already).
    """
    first_slice = (adata.obs["Bregma"] == adata.obs["Bregma"].iloc[0]).to_numpy()
    xy = adata.obsm["spatial"][first_slice]
    spacing = float(np.median(KDTree(xy).query(xy, k=2)[0][:, 1]))  # finding 2 closest points
    # the closest is itself at distance 0, and the second is its real nearest neighbor, so [:, 1]
    # Neighbouring cells are ~5-15 um apart, so a spacing below 0.1 can only mean millimetres.
    if spacing < 0.1:
        adata.obsm["spatial"] = adata.obsm["spatial"] * 1000
        print(f"\nCoordinates are in mm (cells ~{spacing:.4f} apart): converted to um (~{spacing * 1000:.1f} um apart)")
    else:
        print(f"\nCoordinates already look like um (cells ~{spacing:.1f} apart): left as is")


def add_slice_ids(adata):
    """One id per tissue slice. Includes the animal, in case two animals share a Bregma value."""
    # InterScale builds one graph per value of the sample_key column, so each slice needs its own id
    # cells from different slices are never connected
    # adding animal id for safey net
    animal = adata.obs["Animal_ID"].astype(str) if "Animal_ID" in adata.obs else pd.Series("1", index=adata.obs_names)
    bregma = adata.obs["Bregma"].astype(float).map(lambda b: f"{b:+.2f}")
    adata.obs["slice"] = pd.Categorical("animal" + animal + "_bregma" + bregma)


def split_by_slice(adata, held_out_fraction=0.15):
    """Whole slices go to train, val or test. Val and test slices are spread along the brain."""
    # making a list of slices, ordered fron to back
    order = adata.obs.groupby("slice", observed=True)["Bregma"].first().astype(float).sort_values()
    slices = list(order.index)
    n = len(slices)
    # too few slices per split
    if n < 4:
        raise SystemExit(f"Only {n} slices found; at least 4 are needed to split by slice.")
    # how many slices to hold out
    n_held = max(1, round(held_out_fraction * n))  # slices each for val and test
    # Cut the sorted slices into 2 * n_held equal stretches and hold out the middle slice of each,
    # alternating test / val. The two outermost slices always stay in train.
    step = n / (2 * n_held)
    held = [int(step * (j + 0.5)) for j in range(2 * n_held)]
    split = dict.fromkeys(slices, "train")
    for j, i in enumerate(held):
        split[slices[i]] = "test" if j % 2 == 0 else "val"
    # copying the decision onto every cell
    adata.obs["split"] = pd.Categorical(adata.obs["slice"].astype(str).map(split), categories=["train", "val", "test"])

    print(f"\nSplit ({n} slices, sorted by Bregma):")
    for s in slices:
        print(f"  {s:28s} {split[s]:5s} {int((adata.obs['slice'] == s).sum()):6d} cells")
    # this makes a count table, with cell types as rows and splits as columns.
    counts = pd.crosstab(adata.obs[LABEL], adata.obs["split"])
    print("\nCells per class and split:")
    print(counts.to_string())
    missing = counts.index[counts["train"] == 0].tolist()
    if missing:
        print(f"WARNING: these classes never appear in train, so the model cannot learn them: {missing}")


def suggest_radius(adata):
    # InterScale tutorial rule: pick the radius where cells have 10-30 neighbours on average
    compute_neighborhood_stats(adata.copy(), radii=[15, 20, 25, 30, 40], library_key="slice")


def main():
    RAW.parent.mkdir(parents=True, exist_ok=True)
    adata = sq.datasets.merfish(path=RAW)  # downloads the first time, then reads the local copy
    look(adata)
    adata = drop_control_probes(adata)
    adata = clean_cells(adata)
    adata = make_log_scaled(adata)
    coordinates_in_micrometres(adata)
    add_slice_ids(adata)
    split_by_slice(adata)
    suggest_radius(adata)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(OUT)
    print(f"\nSaved {adata.n_obs} cells x {adata.n_vars} genes to {OUT}")


if __name__ == "__main__":
    main()
