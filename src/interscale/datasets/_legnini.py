from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from scanpy import logging as logg
from scipy.sparse import csr_matrix

_ZENODO_URL = "https://zenodo.org/records/8022726/files/merged_transcript_data.csv?download=1"
_CACHE_DIR = Path.home() / ".cache" / "interscale"
_CACHE_FILE = _CACHE_DIR / "legnini_2023.h5ad"


def legnini(path: Path | str | None = None) -> AnnData:
    """Load the Legnini et al. 2023 molecular cartography dataset.

    Legnini, I. et al. Spatiotemporal, optogenetic control of gene expression in organoids. Nat. Methods 20, 1544–1552 (2023).

    Downloads a transcript-level CSV from Zenodo, pivots it to a
    cells-by-genes AnnData object, and caches the result as an .h5ad
    file at ``~/.cache/interscale/legnini_2023.h5ad``.

    On subsequent calls the cached file is returned directly without
    re-downloading.

    Parameters
    ----------
    path
        Optional alternative path for the cached .h5ad file. When
        ``None`` (default), ``~/.cache/interscale/legnini_2023.h5ad``
        is used.

    Returns
    -------
    AnnData
        AnnData object with shape ``(43762, 88)``.

        obs
            ``Cell``, ``Area``, ``x``, ``y``, ``sample``,
            ``condition``, ``organoid``
        var
            ``gene_ids``, ``feature_types``
        obsm
            ``spatial`` — float64 array of shape ``(n_obs, 2)``
        layers
            ``raw`` — raw integer counts
        X
            CSR sparse matrix of raw counts (float32)
    """
    cache_path = Path(path) if path is not None else _CACHE_FILE

    if cache_path.exists():
        import anndata

        logg.info(f"Loading cached dataset from `{cache_path}`")
        return anndata.read_h5ad(cache_path)

    logg.info(f"Downloading Legnini 2023 dataset from `{_ZENODO_URL}`")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    df = _download_csv(_ZENODO_URL)
    adata = _build_adata(df)
    adata.write_h5ad(cache_path)
    logg.info(f"Dataset cached at `{cache_path}`")

    return adata


def _download_csv(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url) as response:
        return pd.read_csv(response)


def _build_adata(df: pd.DataFrame) -> AnnData:
    obs_meta_cols = ["Cell", "Area", "x", "y", "sample", "condition", "organoid"]
    obs_df = df[obs_meta_cols].drop_duplicates(subset="Cell").set_index("Cell")
    obs_df.insert(0, "Cell", obs_df.index)

    counts = df.pivot_table(index="Cell", columns="Gene", values="Counts", aggfunc="sum", fill_value=0)
    counts = counts.loc[obs_df.index]

    X = csr_matrix(counts.values.astype(np.float32))

    var_df = pd.DataFrame(
        {"gene_ids": counts.columns.tolist(), "feature_types": "Gene Expression"},
        index=counts.columns,
    )

    adata = AnnData(X=X, obs=obs_df, var=var_df)
    adata.obsm["spatial"] = obs_df[["x", "y"]].values.astype(np.float64)
    adata.layers["raw"] = csr_matrix(counts.values.astype(np.float32))

    return adata
