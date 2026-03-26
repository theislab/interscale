import os
from typing import Literal, List, Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import rankdata
import seaborn as sns
from sklearn.metrics import r2_score

def predict_gene_r2(adata: AnnData, layers_pred: str, top_n: int = 5) -> pd.DataFrame:
    """
    Predict gene R² scores for a given model layer.
    
    Parameters:
        adata: AnnData object containing the data
        layers_pred: str, name of the model layer to predict
        top_n: int, number of top genes to return
    """
    # Convert y_true to a dense array
    y_true = adata.X.toarray().astype(float)
    
    # Convert predictions to NumPy arrays
    y_pred = adata.layers[layers_pred]
    
    # Ensure predictions are also NumPy arrays (if they're tensors)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    
    # Ensure predictions are also NumPy arrays of type float
    y_pred = y_pred.astype(float)
    
    # Compute R² scores for each gene
    r2_scores = []
    for i in range(y_true.shape[1]):
        # Mask for non-NaN values in both y_true and y_pred for gene i
        mask = ~np.isnan(y_true[:, i]) & ~np.isnan(y_pred[:, i])
        if np.sum(mask) > 1:  # Need at least 2 points to compute R²
            r2 = r2_score(y_true[mask, i], y_pred[mask, i])
        else:
            r2 = np.nan  # Not enough data to compute R²
        r2_scores.append(r2)
    r2_scores_log = [np.log(r2 + 1) for r2 in r2_scores if not np.isnan(r2)]
    r2_ranked = rankdata(r2_scores, method="average")
    
    # Convert to DataFrame for easy sorting
    genes = adata.var_names  # Gene names
    r2_df = pd.DataFrame({'gene': genes, 'r2': r2_scores, 'r2_log': r2_scores_log, 'r2_rank': r2_ranked})
    
    # Get top 5 genes for each model
    top = r2_df.nlargest(top_n, 'r2')
    
    print(f"Top {top_n} genes for {layers_pred} model:\n", top)
    
    return r2_df


def predict_gene_cosine(adata: AnnData, layers_pred: str, top_n: int = 5) -> pd.DataFrame:
    """
    Predict gene cosine similarity scores for a given model layer.
    For each gene, computes cosine similarity between true and predicted expression across cells.

    Parameters:
        adata: AnnData object containing the data
        layers_pred: str, name of the model layer to predict
        top_n: int, number of top genes to return
    """
    y_true = adata.X.toarray().astype(float)
    y_pred = adata.layers[layers_pred]
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    y_pred = y_pred.astype(float)

    cosine_scores = []
    for i in range(y_true.shape[1]):
        mask = ~np.isnan(y_true[:, i]) & ~np.isnan(y_pred[:, i])
        a, b = y_true[mask, i], y_pred[mask, i]
        n = np.sum(mask)
        if n > 0:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a > 0 and norm_b > 0:
                cos_sim = np.dot(a, b) / (norm_a * norm_b)
            else:
                cos_sim = np.nan
        else:
            cos_sim = np.nan
        cosine_scores.append(cos_sim)

    cosine_ranked = rankdata(cosine_scores, method="average")
    genes = adata.var_names
    cosine_df = pd.DataFrame({"gene": genes, "cosine": cosine_scores, "cosine_rank": cosine_ranked})

    top = cosine_df.nlargest(top_n, "cosine")
    print(f"Top {top_n} genes for {layers_pred} model (cosine):\n", top)
    return cosine_df


def gene_rank_analysis(adata,
                       layers_local_pred: str = 'layers_local',
                       layers_global_pred: str = 'layers_global',
                       top_n: int = 5,
                       plot_result: bool = True,
                       return_top_genes: bool = False,
                       save_dir: str = None,
                       post_fix: str = None,
                       score_metric: Literal["r2", "cosine"] = "r2",
                       color_dict: Optional[dict[str, str]] = None):
    """Ranks how well the local and global predictions capture the gene expression.
    Plots the top N predicted genes for each model and consensus genes.

    Args:
        adata: AnnData with layers for local and global predictions.
        layers_local_pred: Layer name for local predictions. Defaults to 'layers_local'.
        layers_global_pred: Layer name for global predictions. Defaults to 'layers_global'.
        top_n: Number of top genes to highlight. Defaults to 5.
        plot_result: Whether to plot. Defaults to True.
        return_top_genes: Whether to return top gene DataFrames. Defaults to False.
        save_dir: Directory to save the figure. If None, figure is not saved. Defaults to None.
        post_fix: Suffix for saved filename. Defaults to None.
        score_metric: 'r2' for R² score, 'cosine' for cosine similarity. Defaults to 'r2'.
        color_dict: Optional mapping for plot colors. Supported keys are
            'all', 'local', 'global', and 'best'. Missing keys use defaults.
    """
    assert layers_local_pred in adata.layers.keys(), f"layers_local_pred {layers_local_pred} not in adata.layers.keys()"
    assert layers_global_pred in adata.layers.keys(), f"layers_global_pred {layers_global_pred} not in adata.layers.keys()"

    if score_metric == "r2":
        local_df = predict_gene_r2(adata, layers_local_pred, top_n)
        global_df = predict_gene_r2(adata, layers_global_pred, top_n)
        rank_col = "r2_rank"
    else:
        local_df = predict_gene_cosine(adata, layers_local_pred, top_n)
        global_df = predict_gene_cosine(adata, layers_global_pred, top_n)
        rank_col = "cosine_rank"

    # Select relevant columns and rename for clarity
    local_df = local_df[["gene", rank_col]].rename(columns={rank_col: "Local Rank"})
    global_df = global_df[["gene", rank_col]].rename(columns={rank_col: "Global Rank"})
    
    # Merge on 'gene'
    merged_df = pd.merge(local_df, global_df, on='gene', how='inner')
    
    # Compute rank difference
    merged_df['Rank Difference'] = merged_df['Local Rank'] - merged_df['Global Rank']
    
    # Compute overall prediction quality (higher avg rank means better prediction)
    merged_df['Avg Rank'] = (merged_df['Local Rank'] + merged_df['Global Rank']) / 2
    
    # Get top_n genes in each category
    top_local_genes = merged_df.nsmallest(top_n, "Rank Difference")  # More local-driven
    top_global_genes = merged_df.nlargest(top_n, "Rank Difference")  # More global-driven
    top_best_genes = merged_df.nlargest(top_n, "Avg Rank")  # Best overall predicted genes
    
    default_colors = {
        "all": "gray",
        "local": "#EE9B00",
        "global": "#27828E",
        "best": "green",
    }
    colors = {**default_colors, **(color_dict or {})}

    # Plot all genes
    plt.figure(figsize=(8, 8))
    plt.scatter(
        merged_df["Local Rank"],
        merged_df["Global Rank"],
        alpha=0.6,
        label="All Genes",
        color=colors["all"],
    )

    # Plot and label top local genes
    plt.scatter(
        top_local_genes["Local Rank"],
        top_local_genes["Global Rank"],
        color=colors["local"],
        label="Top Local",
    )
    for _, row in top_local_genes.iterrows():
        plt.text(
            row["Local Rank"],
            row["Global Rank"],
            row["gene"],
            fontsize=12,
            color=colors["local"],
        )

    # Plot and label top global genes
    plt.scatter(
        top_global_genes["Local Rank"],
        top_global_genes["Global Rank"],
        color=colors["global"],
        label="Top Global",
    )
    for _, row in top_global_genes.iterrows():
        plt.text(
            row["Local Rank"],
            row["Global Rank"],
            row["gene"],
            fontsize=12,
            color=colors["global"],
        )
    
    # Plot and label best-predicted genes
    plt.scatter(
        top_best_genes["Local Rank"],
        top_best_genes["Global Rank"],
        color=colors["best"],
        label="Best Predicted",
    )
    for _, row in top_best_genes.iterrows():
        plt.text(
            row["Local Rank"],
            row["Global Rank"],
            row["gene"],
            fontsize=12,
            color=colors["best"],
        )

    # Reference diagonal
    min_rank, max_rank = merged_df[['Local Rank', 'Global Rank']].values.min(), merged_df[['Local Rank', 'Global Rank']].values.max()
    plt.plot([min_rank, max_rank], [min_rank, max_rank], 'r--', label="Equal Ranking (y=x)")  
    
    # Labels and legend
    plt.xlabel("Local Model Rank")
    plt.ylabel("Global Model Rank")
    plt.title(f"Gene Prediction Rank: Local vs. Global ({score_metric})")
    
    # Save figure if save_dir is provided
    if save_dir is not None:
        name = f"gene_rank_analysis_{score_metric}" + (f"_{post_fix}" if post_fix else "") + ".png"
        save_path = os.path.join(save_dir, name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')    
        print(f"Figure saved to: {save_path}")
    
    plt.show()

    # Return top genes if requested
    if return_top_genes:
        return top_local_genes, top_global_genes, top_best_genes

def _compute_gene_scores(adata: AnnData, layers_pred: str,
                         score_metric: Literal["r2", "cosine"] = "r2") -> pd.DataFrame:
    """Compute per-gene scores (R² or cosine) between adata.X and a prediction layer."""
    y_true = adata.X.toarray().astype(float)
    y_pred = adata.layers[layers_pred]
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    y_pred = y_pred.astype(float)

    scores = []
    for i in range(y_true.shape[1]):
        mask = ~np.isnan(y_true[:, i]) & ~np.isnan(y_pred[:, i])
        a, b = y_true[mask, i], y_pred[mask, i]
        n = np.sum(mask)

        if score_metric == "r2":
            score = r2_score(a, b) if n > 1 else np.nan
        else:  # cosine
            norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
            score = np.dot(a, b) / (norm_a * norm_b) if (n > 0 and norm_a > 0 and norm_b > 0) else np.nan
        scores.append(score)

    ranks = rankdata(scores, method="average")
    return pd.DataFrame({
        "gene": adata.var_names,
        "score": scores,
        "rank": ranks,
    })


def gene_rank_condition_comparison(
    adata: AnnData,
    library_key: str,
    library_id: Optional[str | List[str]] = None,
    layers_local_pred: str = "layers_local",
    layers_global_pred: str = "layers_global",
    top_n: int = 5,
    score_metric: Literal["r2", "cosine"] = "r2",
    color_dict: Optional[dict[str, str]] = None,
    save_dir: Optional[str] = None,
    post_fix: Optional[str] = None,
    return_top_genes: bool = False,
) -> Optional[dict]:
    """Compare gene prediction ranks between two conditions for local and global decoders.

    For each decoder (local / global), genes are scored independently in
    the two selected library IDs. The resulting ranks are plotted against each
    other so you can see which genes are better predicted under one condition
    vs. the other.

    Parameters
    ----------
    adata : AnnData
        Must contain the prediction layers and ``adata.obs[library_key]``.
    library_key : str
        Column in ``adata.obs`` that stores condition / library labels.
    library_id : str or list of str, optional
        Exactly two values from ``adata.obs[library_key]`` to compare.
        If ``None``, the first two unique values in the column are used.
        A single string is interpreted as a one-element list; in that case
        the second unique value is inferred automatically.
    layers_local_pred, layers_global_pred : str
        Layer names for local and global predictions.
    top_n : int
        Number of top genes to highlight per category.
    score_metric : 'r2' | 'cosine'
        Metric used for scoring.
    color_dict : dict, optional
        Mapping of library ID to color string, e.g.
        ``{'ND': '#999999', 'T1D': '#0b559f'}``. If ``None``, defaults
        to blue / red.
    save_dir : str, optional
        Directory for saving figures.
    post_fix : str, optional
        Suffix appended to saved filenames.
    return_top_genes : bool
        If True, return a dict of DataFrames with top genes.

    Returns
    -------
    dict or None
        If ``return_top_genes``, returns a dict with keys
        ``'local'`` and ``'global'``, each containing
        ``(top_id_a, top_id_b, top_consensus)`` DataFrames.
    """
    # --- validate inputs ---
    assert library_key in adata.obs.columns, (
        f"'{library_key}' not found in adata.obs"
    )
    for layer in [layers_local_pred, layers_global_pred]:
        assert layer in adata.layers, f"'{layer}' not found in adata.layers"

    # --- resolve library_id to exactly two values ---
    unique_ids = adata.obs[library_key].unique().tolist()

    if library_id is None:
        assert len(unique_ids) >= 2, (
            f"Need at least 2 unique values in '{library_key}', found {len(unique_ids)}. "
            "Pass library_id explicitly."
        )
        id_a, id_b = unique_ids[0], unique_ids[1]
    else:
        if isinstance(library_id, str):
            library_id = [library_id]
        for lid in library_id:
            assert lid in unique_ids, (
                f"'{lid}' not found in adata.obs['{library_key}']. "
                f"Available: {unique_ids}"
            )
        if len(library_id) == 1:
            remaining = [v for v in unique_ids if v != library_id[0]]
            assert len(remaining) >= 1, (
                f"Only one unique value in '{library_key}'; cannot infer second ID."
            )
            id_a, id_b = library_id[0], remaining[0]
        elif len(library_id) == 2:
            id_a, id_b = library_id[0], library_id[1]
        else:
            raise ValueError(
                f"library_id must contain exactly 1 or 2 values, got {len(library_id)}."
            )

    mask_a = adata.obs[library_key] == id_a
    mask_b = adata.obs[library_key] == id_b
    assert mask_a.sum() > 0, f"No cells found for library_id '{id_a}'"
    assert mask_b.sum() > 0, f"No cells found for library_id '{id_b}'"

    adata_a = adata[mask_a].copy()
    adata_b = adata[mask_b].copy()

    # --- compute scores per condition & decoder ---
    results = {}
    for layer_name, layer_label in [
        (layers_local_pred, "Local"),
        (layers_global_pred, "Global"),
    ]:
        df_a = _compute_gene_scores(adata_a, layer_name, score_metric)
        df_b = _compute_gene_scores(adata_b, layer_name, score_metric)

        merged = pd.merge(
            df_a[["gene", "rank"]].rename(columns={"rank": f"Rank {id_a}"}),
            df_b[["gene", "rank"]].rename(columns={"rank": f"Rank {id_b}"}),
            on="gene",
            how="inner",
        )

        # Rank difference: positive → better in id_b
        merged["Rank Difference"] = (
            merged[f"Rank {id_a}"] - merged[f"Rank {id_b}"]
        )
        merged["Avg Rank"] = (
            merged[f"Rank {id_a}"] + merged[f"Rank {id_b}"]
        ) / 2

        top_a = merged.nlargest(top_n, "Rank Difference")   # better in id_a
        top_b = merged.nsmallest(top_n, "Rank Difference")   # better in id_b
        top_consensus = merged.nlargest(top_n, "Avg Rank")   # best overall

        # --- resolve colors ---
        default_colors = {id_a: "blue", id_b: "red"}
        colors = {**default_colors, **(color_dict or {})}
        color_a = colors[id_a]
        color_b = colors[id_b]

        # --- plot ---
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(
            merged[f"Rank {id_a}"],
            merged[f"Rank {id_b}"],
            alpha=0.5, color="gray",
        )

        for df_top, color in [
            (top_a, color_a),
            (top_b, color_b),
        ]:
            ax.scatter(
                df_top[f"Rank {id_a}"],
                df_top[f"Rank {id_b}"],
                color=color, zorder=3,
            )
            for _, row in df_top.iterrows():
                ax.annotate(
                    row["gene"],
                    xy=(row[f"Rank {id_a}"], row[f"Rank {id_b}"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=12,
                    color=color,
                    fontweight="bold",
                )

        # diagonal
        lo = merged[[f"Rank {id_a}", f"Rank {id_b}"]].values.min()
        hi = merged[[f"Rank {id_a}", f"Rank {id_b}"]].values.max()
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4)

        ax.set_xlabel(f"Rank — {id_a}")
        ax.set_ylabel(f"Rank — {id_b}")
        ax.set_title(
            f"{layer_label} Decoder: {id_a} vs {id_b} ({score_metric})"
        )

        # legend with only condition colors, placed outside
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color_a,
                   markersize=8, label=id_a),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color_b,
                   markersize=8, label=id_b),
        ]
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1),
                  loc="upper left", borderaxespad=0, frameon=False)

        if save_dir is not None:
            name = (
                f"gene_rank_{layer_label.lower()}_{id_a}_vs_{id_b}_{score_metric}"
                + (f"_{post_fix}" if post_fix else "")
                + ".png"
            )
            path = os.path.join(save_dir, name)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {path}")

        plt.show()

        results[layer_label.lower()] = (top_a, top_b, top_consensus)

    if return_top_genes:
        return results