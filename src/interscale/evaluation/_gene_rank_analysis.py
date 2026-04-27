import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import rankdata
from sklearn.metrics import r2_score


def _predict_gene_r2(adata: AnnData, layers_pred: str) -> pd.DataFrame:
    """
    Predict gene R2 scores for a given model layer.

    Parameters
    ----------
        adata: AnnData object containing the data
        layers_pred: str, name of the model layer to predict

    Returns
    -------
        pd.DataFrame
            DataFrame with columns: gene, r2, r2_log, r2_rank for all genes
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
    # r2_scores_log = [np.log(r2 + 1) for r2 in r2_scores if not np.isnan(r2)]
    r2_scores_log = [np.log1p(r2) if not np.isnan(r2) else np.nan for r2 in r2_scores]
    r2_ranked = rankdata(r2_scores, method="average")

    # Convert to DataFrame for easy sorting
    genes = adata.var_names  # Gene names
    r2_df = pd.DataFrame({"gene": genes, "r2": r2_scores, "r2_log": r2_scores_log, "r2_rank": r2_ranked})

    return r2_df


def calculate_gene_ranks(
    adata: AnnData,
    *,
    layers_local_pred: str = "layers_local",
    layers_global_pred: str = "layers_global",
    top_n: int = None,
) -> pd.DataFrame:
    """Calculate gene ranks comparing local and global model predictions.

    Computes R² scores for both local and global predictions, merges the results,
    and calculates ranking statistics. Optionally prints top genes.

    Parameters
    ----------
        adata : AnnData
            AnnData object containing the data and prediction layers
        layers_local_pred : str, optional
            Name of the local model prediction layer. Defaults to 'layers_local'.
        layers_global_pred : str, optional
            Name of the global model prediction layer. Defaults to 'layers_global'.
        top_n : int, optional
            Number of top genes to print for each category. If None, no printing occurs.
            Defaults to None.

    Returns
    -------
        pd.DataFrame
            DataFrame with columns: gene, Local Rank, Global Rank, Rank Difference, Avg Rank
    """
    assert layers_local_pred in adata.layers.keys(), f"layers_local_pred {layers_local_pred} not in adata.layers.keys()"
    assert layers_global_pred in adata.layers.keys(), (
        f"layers_global_pred {layers_global_pred} not in adata.layers.keys()"
    )

    local_df = _predict_gene_r2(adata, layers_local_pred)
    global_df = _predict_gene_r2(adata, layers_global_pred)

    # Select relevant columns and rename for clarity
    local_df = local_df[["gene", "r2_rank"]].rename(columns={"r2_rank": "Local Rank"})
    global_df = global_df[["gene", "r2_rank"]].rename(columns={"r2_rank": "Global Rank"})

    # Merge on 'gene'
    merged_df = pd.merge(local_df, global_df, on="gene", how="inner")

    # Compute rank difference
    merged_df["Rank Difference"] = merged_df["Local Rank"] - merged_df["Global Rank"]

    # Compute overall prediction quality (higher avg rank means better prediction)
    merged_df["Avg Rank"] = (merged_df["Local Rank"] + merged_df["Global Rank"]) / 2

    # Print top genes if top_n is specified
    if top_n is not None:
        top_local_genes = merged_df.nsmallest(top_n, "Rank Difference")  # More local-driven
        top_global_genes = merged_df.nlargest(top_n, "Rank Difference")  # More global-driven
        top_best_genes = merged_df.nlargest(top_n, "Avg Rank")  # Best overall predicted genes

        print(f"\nTop {top_n} Local-Driven Genes:\n", top_local_genes[["gene", "Local Rank", "Global Rank"]])
        print(f"\nTop {top_n} Global-Driven Genes:\n", top_global_genes[["gene", "Local Rank", "Global Rank"]])
        print(f"\nTop {top_n} Best Predicted Genes:\n", top_best_genes[["gene", "Local Rank", "Global Rank"]])

    return merged_df
