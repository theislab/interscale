import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from interscale.evaluation import _get_Z


def latent_correlation(
    adata,
    z_key="_global_emb",
    vmax=1.0,
    cmap="BrBG_r",
    figsize=(10, 10),
    label_fontsize=8,
    title=None,
    method="average",
    metric="correlation",
    show=True,
):
    """
    Clustermap of Corr(Z) for embedding dimensions.
    """
    Z = _get_Z(adata, z_key)
    C = np.corrcoef(Z, rowvar=False)

    labels = [f"{i}" for i in range(C.shape[0])]

    g = sns.clustermap(
        C,
        dendrogram_ratio=(0.08, 0.08),
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        figsize=figsize,
        method=method,
        metric=metric,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "correlation"},
    )

    # Rotate and resize tick labels
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=label_fontsize)

    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=label_fontsize)

    g.ax_heatmap.set_xlabel("latent dim")
    g.ax_heatmap.set_ylabel("latent dim")
    # g.ax_heatmap.set_title(title or f"Correlation of {z_key}", pad=12)

    plt.tight_layout()

    if show:
        plt.show()

    return g


def dim_importance_elbow_stdexpr(
    adata,
    s_key="_global_std_gene_loadings",
    z_key="_global_emb",
    mode="full",
    use_ratio=True,
    cumulative_cutoff=0.90,
    spacing=2,
    n_top=None,  # NEW
    figsize=(8, 4.5),
    fontsize=12,
    title=None,
    show=True,
):

    if s_key not in adata.varm:
        raise KeyError(f"{s_key} not found in adata.varm")
    if z_key not in adata.obsm:
        raise KeyError(f"{z_key} not found in adata.obsm")

    S = np.asarray(adata.varm[s_key], dtype=float)
    Z = np.asarray(adata.obsm[z_key], dtype=float)

    if Z.ndim == 1:
        Z = Z[:, None]

    if S.ndim != 2 or S.shape[1] != Z.shape[1]:
        raise ValueError(f"Shape mismatch: S is {S.shape}, Z is {Z.shape}")

    ok = np.isfinite(Z).all(axis=1)
    if ok.sum() < 3:
        raise ValueError(f"Need >=3 finite rows in adata.obsm['{z_key}']")

    Z = Z[ok]

    # -------------------
    # scoring
    # -------------------

    if mode == "diag":
        score = np.sum(S * S, axis=0)

    elif mode == "full":
        Corr = np.corrcoef(Z, rowvar=False)
        StS = S.T @ S
        A = StS * Corr
        A = 0.5 * (A + A.T)
        score = A.sum(axis=0)

    else:
        raise ValueError("mode must be 'diag' or 'full'")

    # -------------------
    # y values
    # -------------------

    if use_ratio:
        total = float(np.sum(score))
        y = score / total if total > 0 else np.zeros_like(score)
        ylab = "Importance ratio (std-expr)"
    else:
        y = score
        ylab = "Importance (std-expr)"

    # -------------------
    # sorting
    # -------------------

    order = np.argsort(y)[::-1]
    y_sorted = y[order]
    dim_sorted = order

    # -------------------
    # cutoff
    # -------------------

    cutoff_x = None
    cutoff_idx = None

    if use_ratio and cumulative_cutoff is not None and y.sum() > 0:
        cum = np.cumsum(y_sorted)
        cutoff_idx = int(np.searchsorted(cum, cumulative_cutoff, side="left"))
        cutoff_x = cutoff_idx * spacing

        n_dims_left = cutoff_idx + 1
        dims_left = dim_sorted[:n_dims_left]

        print(f"\n{int(cumulative_cutoff * 100)}% cumulative variance reached with {n_dims_left} dimensions:")
        print("Embedding dimensions (sorted by importance):")
        print(dims_left.tolist())

    # -------------------
    # apply n_top filter
    # -------------------

    K = len(y_sorted)

    if n_top is not None:
        K = min(n_top, K)

    y_plot = y_sorted[:K]
    dim_plot = dim_sorted[:K]
    x = np.arange(K) * spacing

    # -------------------
    # plotting
    # -------------------

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, y_plot, marker="o", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(dim_plot.astype(str), rotation=45, ha="right", fontsize=fontsize - 1)

    # cutoff line only if visible
    if cutoff_x is not None and cutoff_idx < K:
        ax.axvline(
            x=cutoff_x,
            linestyle="--",
            linewidth=1,
            color="red",
            label=f"{int(cumulative_cutoff * 100)}% cumulative",
        )
        ax.legend(fontsize=fontsize - 2, loc="best")

    ax.set_title(
        title or f"Elbow (std-expr, {mode}): {s_key} + Corr({z_key})",
        fontsize=fontsize + 1,
    )

    ax.set_xlabel("Embedding dim (sorted)", fontsize=fontsize)
    ax.set_ylabel(ylab, fontsize=fontsize)

    ax.tick_params(axis="both", labelsize=fontsize - 1)
    ax.grid(False)

    plt.tight_layout()

    if show:
        plt.show()

    return ax


def gene_ranks(
    merged_df: pd.DataFrame,
    *,
    top_n: int = 5,
    save_dir: str = None,
    post_fix: str = None,
    color_local: str = "EE9B00",
    color_global: str = "005F73",
):
    """Plot gene ranks comparing local and global model predictions.

    Creates a scatter plot visualizing the ranking relationships between local and global
    model predictions, highlighting top genes in various categories.

    Parameters
    ----------
        merged_df : pd.DataFrame
            DataFrame with columns: gene, Local Rank, Global Rank, Rank Difference, Avg Rank
        top_n : int, optional
            Number of top genes to highlight in each category. Defaults to 5.
        save_dir : str, optional
            Directory to save the figure. If None, figure is not saved. Defaults to None.
        post_fix : str, optional
            Post-fix to append to the saved figure filename. Defaults to None.
        color_local : str, optional
            Hex color code for local-driven genes. Defaults to 'EE9B00'.
        color_global : str, optional
            Hex color code for global-driven genes. Defaults to '005F73'.
    """
    # Get top_n genes in each category
    top_local_genes = merged_df.nsmallest(top_n, "Rank Difference")  # More local-driven
    top_global_genes = merged_df.nlargest(top_n, "Rank Difference")  # More global-driven
    top_best_genes = merged_df.nlargest(top_n, "Avg Rank")  # Best overall predicted genes

    # Ensure colors have '#' prefix
    if not color_local.startswith("#"):
        color_local = f"#{color_local}"
    if not color_global.startswith("#"):
        color_global = f"#{color_global}"

    # Plot all genes
    plt.figure(figsize=(8, 8))
    plt.scatter(merged_df["Local Rank"], merged_df["Global Rank"], alpha=0.6, label="All Genes", color="gray")

    # Plot and label top local genes
    plt.scatter(top_local_genes["Local Rank"], top_local_genes["Global Rank"], color=color_local, label="Top Local")
    for _, row in top_local_genes.iterrows():
        plt.text(row["Local Rank"], row["Global Rank"], row["gene"], fontsize=10, color=color_local)

    # Plot and label top global genes
    plt.scatter(top_global_genes["Local Rank"], top_global_genes["Global Rank"], color=color_global, label="Top Global")
    for _, row in top_global_genes.iterrows():
        plt.text(row["Local Rank"], row["Global Rank"], row["gene"], fontsize=10, color=color_global)

    # Plot and label best-predicted genes
    plt.scatter(top_best_genes["Local Rank"], top_best_genes["Global Rank"], color="green", label="Best Predicted")
    for _, row in top_best_genes.iterrows():
        plt.text(row["Local Rank"], row["Global Rank"], row["gene"], fontsize=10, color="green")

    # Reference diagonal
    min_rank, max_rank = (
        merged_df[["Local Rank", "Global Rank"]].values.min(),
        merged_df[["Local Rank", "Global Rank"]].values.max(),
    )
    plt.plot([min_rank, max_rank], [min_rank, max_rank], "r--", label="Equal Ranking (y=x)")

    # Labels and legend
    plt.xlabel("Local Model Rank")
    plt.ylabel("Global Model Rank")
    plt.title("Gene Prediction Rank: Local vs. Global")
    plt.legend()

    # Save figure if save_dir is provided
    if save_dir is not None:
        save_path = os.path.join(save_dir, f"gene_rank_analysis_{post_fix}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()
