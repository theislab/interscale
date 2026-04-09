import math
from collections.abc import Sequence

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


def gene_set_covariance(
    adata: ad.AnnData,
    gene_sets: dict,
    varm_key: str,
    plot: bool | Sequence[str] = False,  # now accepts list of names
    variance_cutoff: float = 0.6,
    ncols: int = 3,
    fontsize: int = 8,
) -> pd.DataFrame:
    """
    Computes the diagonal of the covariance matrix (variance) per embedding dimension
    for each gene set using an AnnData object. Optionally generates elbow plots.

    Parameters
    ----------
    - adata: AnnData object containing gene embeddings in `adata.varm[varm_key]`.
    - gene_sets: Dict of gene sets {set_name: list of gene names}.
    - varm_key: Key in `adata.varm` where the embedding matrix is stored.
    - plot:
        * False -> no plots (default)
        * True  -> plot all gene sets
        * Sequence[str] -> plot only these gene-set names (intersection with available)
    - variance_cutoff: Cumulative variance threshold (0 < value <= 1) to mark in plot.
    - ncols: Number of columns in the subplot grid when plotting.
    - fontsize: Base font size for labels/ticks/titles.

    Returns
    -------
    - DataFrame with MultiIndex (gene_set, embedding_dimension) and covariance values.
    """
    df = pd.DataFrame(adata.varm[varm_key], index=adata.var_names)

    # Determine which sets to plot
    if isinstance(plot, (list, tuple, set)):
        requested = set(plot)
        available = set(gene_sets.keys())
        plot_only = requested & available
        do_plot = len(plot_only) > 0
    else:
        plot_only = set(gene_sets.keys()) if plot is True else set()
        do_plot = bool(plot)

    all_results = []
    plot_payload = []  # collect per-set data for grid plotting

    for set_name, genes in gene_sets.items():
        # Skip plotting if a subset was requested and this set isn't in it
        will_plot_this = (set_name in plot_only) if do_plot and len(plot_only) > 0 else (plot is True)

        filtered_genes = [gene for gene in genes if gene in df.index]
        if len(filtered_genes) < 2:
            continue  # need at least 2 genes to compute covariance

        subset = df.loc[filtered_genes]
        cov_matrix = subset.cov()
        diagonal_cov = np.diag(cov_matrix)

        temp_df = pd.DataFrame(
            {"gene_set": set_name, "embedding_dimension": cov_matrix.columns, "covariance": diagonal_cov}
        )

        # Sort by covariance descending and calculate cumulative percentage
        sorted_df = temp_df.sort_values(by="covariance", ascending=False).reset_index(drop=True)
        sorted_df["embedding_index"] = sorted_df["embedding_dimension"]
        sorted_df["cum_sum"] = sorted_df["covariance"].cumsum()
        total = sorted_df["covariance"].sum()

        if total > 0:
            sorted_df["cum_percent"] = sorted_df["cum_sum"] / total
            cutoff_idx = (sorted_df["cum_percent"] <= variance_cutoff).sum() - 1
            cutoff_dim = cutoff_idx if cutoff_idx >= 0 else 0
        else:
            sorted_df["cum_percent"] = 0.0
            cutoff_dim = 0

        all_results.append(temp_df)

        if do_plot and will_plot_this:
            plot_payload.append({"set_name": set_name, "sorted_df": sorted_df, "cutoff_dim": cutoff_dim})

    if do_plot and plot_payload:
        nplots = len(plot_payload)
        nrows = math.ceil(nplots / ncols)

        # Figure size scales with grid
        fig_w = ncols * 7
        fig_h = nrows * 4
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

        for i, payload in enumerate(plot_payload):
            r, c = divmod(i, ncols)
            ax = axes[r, c]
            sd = payload["sorted_df"]
            cutoff_dim = payload["cutoff_dim"]
            set_name = payload["set_name"]

            y = sd["covariance"].values
            x = np.arange(len(sd)) * 2  # spacing between dots

            ax.plot(x, y, marker="o", linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels(sd["embedding_index"], rotation=45, ha="right", fontsize=fontsize - 1)
            ax.axvline(
                x=cutoff_dim * 2,
                linestyle="--",
                linewidth=1,
                color="red",
                label=f"{int(variance_cutoff * 100)}% variance",
            )

            ax.set_title(set_name, fontsize=fontsize + 1)
            ax.set_xlabel("Embedding dim (sorted by variance)", fontsize=fontsize)
            ax.set_ylabel("Variance", fontsize=fontsize)
            ax.tick_params(axis="both", labelsize=fontsize - 1)
            ax.grid(False)
            ax.legend(fontsize=fontsize - 2, loc="best")

        # Hide leftover axes
        total_axes = nrows * ncols
        for j in range(nplots, total_axes):
            r, c = divmod(j, ncols)
            axes[r, c].axis("off")

        fig.suptitle("Elbow plots per gene set", fontsize=fontsize + 3)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()

    if not all_results:
        return pd.DataFrame(columns=["covariance"]).astype({"covariance": float})

    return pd.concat(all_results).set_index(["gene_set", "embedding_dimension"])


def spatial_covariance_plot(
    adata,
    gene_sets,
    int_length="local",  # "local" or "global"
    restrict_to=None,
    emb_dim=None,  # if None, choose highest-covariance dim per set
    cmap="plasma",  # colormap for plotting
):
    """
    For each (selected) gene set, plot one embedding dimension on the spatial embedding.

    Parameters
    ----------
    adata : AnnData
    gene_sets : dict
        Dictionary mapping gene set names to lists of gene IDs.
    int_length : {"local", "global"}
        Which embedding to use.
    restrict_to : list, optional
        Subset of gene set names to plot. Default = all sets.
    emb_dim : int, optional
        Dimension to plot. If None, pick the dimension with the highest
        covariance for each gene set.
    cmap : str, optional
        Matplotlib colormap name to use for plotting (default: "viridis").

    Returns
    -------
    dims_used : dict
        Mapping from gene set name to the dimension that was plotted.
    """
    int_length = str(int_length).lower()
    key_map = {
        "local": {"varm_key": "_local_std_gene_loadings", "obsm_key": "_local_emb", "title_tag": "Local Emb"},
        "global": {"varm_key": "_global_std_gene_loadings", "obsm_key": "_global_emb", "title_tag": "Global Emb"},
    }
    if int_length not in key_map:
        raise ValueError("int_length must be 'local' or 'global'.")

    varm_key = key_map[int_length]["varm_key"]
    obsm_key = key_map[int_length]["obsm_key"]
    title_tag = key_map[int_length]["title_tag"]

    if varm_key not in adata.varm:
        raise KeyError(f"{varm_key!r} not found in adata.varm.")
    if obsm_key not in adata.obsm:
        raise KeyError(f"{obsm_key!r} not found in adata.obsm.")

    # Build dataframes for convenience
    decoder_df = pd.DataFrame(adata.varm[varm_key], index=adata.var_names)
    local_or_global_emb = pd.DataFrame(adata.obsm[obsm_key], index=adata.obs_names)
    n_dims = local_or_global_emb.shape[1]

    # Validate emb_dim
    if emb_dim is not None:
        if not isinstance(emb_dim, int):
            raise TypeError("emb_dim must be an integer or None.")
        if emb_dim < 0 or emb_dim >= n_dims:
            raise ValueError(f"emb_dim={emb_dim} is out of bounds for embedding with {n_dims} dimensions.")

    # Subset gene sets if restrict_to is given
    if restrict_to is not None:
        gene_sets = {name: genes for name, genes in gene_sets.items() if name in restrict_to}
        if len(gene_sets) == 0:
            raise ValueError("No gene set names in 'restrict_to' matched keys in 'gene_sets'.")

    dims_used = {}

    for set_name, genes in gene_sets.items():
        filtered_genes = [g for g in genes if g in decoder_df.index]
        if len(filtered_genes) < 2:
            continue

        subset = decoder_df.loc[filtered_genes]
        cov_matrix = subset.cov()

        diag_var = np.diag(cov_matrix.values)
        default_dim = int(np.argmax(diag_var))

        dim_to_plot = emb_dim if emb_dim is not None else default_dim
        if dim_to_plot >= n_dims:
            continue

        dims_used[set_name] = dim_to_plot

        obs_key = f"{set_name}_{int_length}"
        adata.obs[obs_key] = local_or_global_emb.iloc[:, dim_to_plot].values

        sc.pl.embedding(
            adata,
            basis="spatial",
            color=obs_key,
            title=f"Gene set: {set_name} - {title_tag} (dim {dim_to_plot})",
            cmap=cmap,
            show=True,
        )

    return dims_used
