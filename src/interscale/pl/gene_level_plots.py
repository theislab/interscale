import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from interscale.evaluation import _get_Z
from interscale.pl.config import Plotting


def latent_correlation(
    adata,
    *,
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


def dim_importance_elbow(
    results,
    figsize=(8, 4.5),
    fontsize=12,
    title=None,
    show=True,
    ax=None,
):
    """Plot dimension importance elbow plot with cumulative cutoff.

    Parameters
    ----------
    results : dict
        Dictionary returned from calculate_dim_importance().
    figsize : tuple
        Figure size (only used if ax is None).
    fontsize : int
        Font size for labels.
    title : str, optional
        Plot title. If None, auto-generated.
    show : bool
        Whether to show the plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object.
    """
    y_plot = results["y_plot"]
    dim_plot = results["dim_plot"]
    cutoff_x = results["cutoff_x"]
    cutoff_idx = results["cutoff_idx"]
    n_dims_left = results["n_dims_left"]
    dims_left = results["dims_left"]
    cumulative_cutoff = results["cumulative_cutoff"]
    mode = results["mode"]
    s_key = results["s_key"]
    z_key = results["z_key"]
    use_ratio = results["use_ratio"]
    spacing = results["spacing"]

    ylab = "Importance ratio (std-expr)" if use_ratio else "Importance (std-expr)"

    K = len(y_plot)
    x = np.arange(K) * spacing

    if ax is None:
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

    if title is None:
        title = f"Elbow (std-expr, {mode}): {s_key} + Corr({z_key})"

    ax.set_title(title, fontsize=fontsize + 1)
    ax.set_xlabel("Embedding dim (sorted)", fontsize=fontsize)
    ax.set_ylabel(ylab, fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize - 1)
    ax.grid(False)

    if ax is None:
        plt.tight_layout()

    if show and ax is None:
        plt.show()

    # Print cutoff info
    if n_dims_left is not None:
        print(f"\n{int(cumulative_cutoff * 100)}% cumulative variance reached with {n_dims_left} dimensions:")
        print("Embedding dimensions (sorted by importance):")
        print(dims_left.tolist())

    return ax


def gene_ranks(
    merged_df: pd.DataFrame,
    *,
    top_n: int = 5,
    save_dir: str = None,
    post_fix: str = None,
    color_local: str = None,
    color_global: str = None,
    plotting_config: Plotting = None,
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
            Hex color code for local-driven genes. If None, uses default from plotting config.
        color_global : str, optional
            Hex color code for global-driven genes. If None, uses default from plotting config.
        plotting_config : Plotting, optional
            Plotting configuration object. If None, uses default configuration.
    """
    # Initialize plotting config if not provided
    if plotting_config is None:
        plotting_config = Plotting()

    # Get plotting parameters from config
    general_cfg = plotting_config.config["plot_configs"]["general"]
    rank_cfg = plotting_config.config["plot_configs"].get("rank_genes_plots", {})

    figsize = (10, 10)
    fontsize = general_cfg.get("title_fontsize", 14)
    legend_fontsize = general_cfg.get("legend_fontsize", 12)
    dpi_save = general_cfg.get("dpi_save", 300)

    # Use defaults if colors not provided
    if color_local is None:
        color_local = rank_cfg.get("color_local", "EE9B00")
    if color_global is None:
        color_global = rank_cfg.get("color_global", "005F73")

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
    plt.figure(figsize=figsize)
    plt.scatter(merged_df["Local Rank"], merged_df["Global Rank"], alpha=0.6, label="All Genes", color="gray")

    # Plot and label top local genes
    plt.scatter(top_local_genes["Local Rank"], top_local_genes["Global Rank"], color=color_local, label="Top Local")
    for _, row in top_local_genes.iterrows():
        plt.text(row["Local Rank"], row["Global Rank"], row["gene"], fontsize=fontsize - 4, color=color_local)

    # Plot and label top global genes
    plt.scatter(top_global_genes["Local Rank"], top_global_genes["Global Rank"], color=color_global, label="Top Global")
    for _, row in top_global_genes.iterrows():
        plt.text(row["Local Rank"], row["Global Rank"], row["gene"], fontsize=fontsize - 4, color=color_global)

    # Plot and label best-predicted genes
    plt.scatter(top_best_genes["Local Rank"], top_best_genes["Global Rank"], color="green", label="Best Predicted")
    for _, row in top_best_genes.iterrows():
        plt.text(row["Local Rank"], row["Global Rank"], row["gene"], fontsize=fontsize - 4, color="green")

    # Reference diagonal
    min_rank, max_rank = (
        merged_df[["Local Rank", "Global Rank"]].values.min(),
        merged_df[["Local Rank", "Global Rank"]].values.max(),
    )
    plt.plot([min_rank, max_rank], [min_rank, max_rank], "r--", label="Equal Ranking (y=x)")

    # Labels and legend
    plt.xlabel("Local Model Rank", fontsize=fontsize)
    plt.ylabel("Global Model Rank", fontsize=fontsize)
    plt.title(
        "Gene Prediction Rank: Local vs. Global",
        fontsize=fontsize,
        fontweight=general_cfg.get("title_fontweight", "bold"),
    )
    plt.legend(fontsize=legend_fontsize)

    # Save figure if save_dir is provided
    if save_dir is not None:
        save_path = os.path.join(save_dir, f"gene_rank_analysis_{post_fix}.png")
        plt.savefig(save_path, dpi=dpi_save, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()
