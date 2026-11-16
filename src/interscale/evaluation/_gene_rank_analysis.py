import numpy as np
import pandas as pd
from anndata import AnnData
import torch
from scipy.sparse import issparse
from scipy.stats import rankdata

from interscale.train._trainingplans import _MASKED_MOMENTS, _masked_moments, per_gene_metrics_from_moments

GENE_RANK_METRICS = ("r2", "pearson_corr", "concordance_corr", "cosine_similarity")
_CHUNK_SIZE = 50_000


def _dense(x) -> np.ndarray:
    return (x.toarray() if issparse(x) else np.asarray(x)).astype(float)


def _predict_gene_metric(
    adata: AnnData, layer_key: str | None, *, layers_pred: str, metric: str = "r2"
) -> pd.DataFrame:
    """Score every gene with one of the metrics used for model selection.

    The score comes from the same per-gene moments as the validation metrics logged in training
    (``interscale.train._trainingplans``), so ``r2``, ``pearson_corr`` and ``concordance_corr``
    match training. ``cosine_similarity`` is the cosine over the cells of one gene; training
    logs the cosine over the genes of one cell, which has no per-gene value to rank.

    Parameters
    ----------
    adata
        AnnData object containing the data and the prediction layer.
    layer_key
        Layer with the normalized expression values. If None, ``adata.X``.
    layers_pred
        Name of the layer with the model predictions.
    metric
        One of ``GENE_RANK_METRICS``.

    Returns
    -------
    pd.DataFrame
        Columns ``gene``, ``score``, ``rank``. A higher rank means a better score; genes with an
        undefined score (NaN) get the lowest ranks.
    """
    if metric not in GENE_RANK_METRICS:
        raise ValueError(f"metric must be one of {GENE_RANK_METRICS}, got {metric!r}")

    y_true_all = adata.X if layer_key is None else adata.layers[layer_key]
    y_pred_all = adata.layers[layers_pred]

    # Every moment is a plain sum, so moments of row chunks add up to the moments of all cells.
    moments = None
    max_true, integer_valued = -np.inf, True
    for start in range(0, adata.n_obs, _CHUNK_SIZE):
        y_true = _dense(y_true_all[start : start + _CHUNK_SIZE])
        y_pred = _dense(y_pred_all[start : start + _CHUNK_SIZE])
        valid = np.isfinite(y_true) & np.isfinite(y_pred)

        finite_true = y_true[np.isfinite(y_true)]
        if finite_true.size:
            max_true = max(max_true, float(finite_true.max()))
            integer_valued &= bool(np.allclose(finite_true, np.round(finite_true)))

        chunk = _masked_moments(
            torch.from_numpy(np.where(valid, y_pred, 0.0)),
            torch.from_numpy(np.where(valid, y_true, 0.0)),
            torch.from_numpy(valid),
        )
        moments = chunk if moments is None else {k: moments[k] + chunk[k] for k in _MASKED_MOMENTS}

    # ensure y_true is on same scale as during model training (e.g. log1p normalized) and not raw counts
    looks_like_counts = integer_valued and max_true > 1
    assert max_true <= 20 and not looks_like_counts, (
        f"adata.X looks like raw or normalized counts (max={max_true:.3g}"
        f"{', integer-valued' if looks_like_counts else ''}). Please pass layer_key that contains the normalized counts."
    )

    scores = per_gene_metrics_from_moments(moments)[metric].numpy()
    # rankdata propagates NaN to every gene, so genes without a defined score rank lowest instead.
    ranks = rankdata(np.where(np.isnan(scores), -np.inf, scores), method="average")
    return pd.DataFrame({"gene": adata.var_names, "score": scores, "rank": ranks})



def calculate_gene_ranks(
    adata: AnnData,
    *,
    layer_key: str | None = None,
    layers_local_pred: str = "layers_local",
    layers_global_pred: str = "layers_global",
    top_n: int = None,
    metric: str = "r2",
) -> pd.DataFrame:
    """Calculate gene ranks comparing local and global model predictions.

    Scores every gene per prediction layer with ``metric``, ranks the genes, merges the
    results, and calculates ranking statistics. Optionally prints top genes.

    Parameters
    ----------
        adata : AnnData
            AnnData object containing the data and prediction layers
        layer_key: str, optional
            Name of layer in adata that contains the normalized GEX values. If None, then X. 
        layers_local_pred : str, optional
            Name of the local model prediction layer. Defaults to 'layers_local'.
        layers_global_pred : str, optional
            Name of the global model prediction layer. Defaults to 'layers_global'.
        top_n : int, optional
            Number of top genes to print for each category. If None, no printing occurs.
            Defaults to None.
        metric : str, optional
            Per-gene metric to rank by: ``r2``, ``pearson_corr``, ``concordance_corr`` or
            ``cosine_similarity``, the names of the validation metrics logged in training. Use the
            one that selected the model. Defaults to ``r2``. ``cosine_similarity`` is the cosine
            over the cells of one gene; training logs it over the genes of one cell.

    Returns
    -------
        pd.DataFrame
            DataFrame with columns: gene, Local Rank, Global Rank, Rank Difference, Avg Rank
    """
    assert layers_local_pred in adata.layers.keys(), f"layers_local_pred {layers_local_pred} not in adata.layers.keys()"
    assert layers_global_pred in adata.layers.keys(), (
        f"layers_global_pred {layers_global_pred} not in adata.layers.keys()"
    )

    local_df = _predict_gene_metric(adata, layer_key, layers_pred=layers_local_pred, metric=metric)
    global_df = _predict_gene_metric(adata, layer_key, layers_pred=layers_global_pred, metric=metric)

    # Select relevant columns and rename for clarity
    local_df = local_df[["gene", "rank"]].rename(columns={"rank": "Local Rank"})
    global_df = global_df[["gene", "rank"]].rename(columns={"rank": "Global Rank"})

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
