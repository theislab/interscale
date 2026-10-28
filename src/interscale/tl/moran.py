"""Spatial scale-dependent Moran analysis.

Moran's I of a feature (gene or latent dimension) is computed on a series of spatial kNN graphs with
increasing ``k`` (or radius graphs with increasing radius). The resulting profile ``I(k)`` describes how far
spatial autocorrelation persists: localized patterns decay quickly with ``k``, tissue-scale patterns stay high.
The profile is summarized by its area under the curve (trapezoidal rule over ``k`` or radius) and its decay is split
into a local (small scales) and a global (large scales) part.

    moran_scale(adata, ks=(4, 8, 16, 32, 64, 128), library_key="sample", layer="log1p_norm", per_library=True)
    adata.uns["moran_scale"]["stats"]  # AUC, local / global decay and local index per gene

Plots of the results are in ``pl/moran_plots.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import squidpy as sq
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors

__all__ = ["DEFAULT_KS", "moran_scale", "moran_profile", "auc_from_profile", "delta_from_profile", "radii_from_knn"]

DEFAULT_KS = (4, 8, 16, 32, 64)


def _feature_adata(
    adata: ad.AnnData,
    genes: Sequence[str] | None,
    layer: str | None,
    obsm_key: str | None,
    library_key: str | None,
    spatial_key: str,
) -> ad.AnnData:
    """Lightweight AnnData holding only the features, coordinates and library labels."""
    if obsm_key is not None:
        X = np.asarray(adata.obsm[obsm_key])
        var_names = [f"{obsm_key}_{i}" for i in range(X.shape[1])]
    else:
        genes = list(adata.var_names) if genes is None else list(genes)
        sub = adata[:, genes]
        X = sub.X if layer is None else sub.layers[layer]
        X = X.toarray() if issparse(X) else np.asarray(X)
        var_names = genes
    obs = adata.obs[[library_key]].copy() if library_key is not None else pd.DataFrame(index=adata.obs_names)
    fa = ad.AnnData(X=X.astype(np.float64), obs=obs, var=pd.DataFrame(index=var_names))
    fa.obsm["spatial"] = np.asarray(adata.obsm[spatial_key])
    return fa


@contextmanager
def _quiet_graph_log() -> Iterator[None]:
    """Silence squidpy's per-graph INFO log ("Creating graph using ..."); warnings still show.

    squidpy's graph builder logs through the ``spatialdata._logging`` logger (with its own level set), not scanpy's,
    so scanpy's verbosity has no effect.
    """
    logger = logging.getLogger("spatialdata._logging")
    level = logger.level
    logger.setLevel(max(level, logging.WARNING))
    try:
        yield
    finally:
        logger.setLevel(level)


def _spatial_graph(fa: ad.AnnData, scale: float, graph: str, library_key: str | None) -> None:
    # `spatial_neighbors_knn/radius` replace the deprecated `spatial_neighbors(coord_type="generic", ...)`.
    if graph == "knn":
        if hasattr(sq.gr, "spatial_neighbors_knn"):
            sq.gr.spatial_neighbors_knn(fa, n_neighs=int(scale), library_key=library_key)
        else:
            sq.gr.spatial_neighbors(fa, coord_type="generic", n_neighs=int(scale), library_key=library_key)
    elif graph == "radius":
        if hasattr(sq.gr, "spatial_neighbors_radius"):
            sq.gr.spatial_neighbors_radius(fa, radius=float(scale), library_key=library_key)
        else:
            sq.gr.spatial_neighbors(fa, coord_type="generic", radius=float(scale), library_key=library_key)
    else:
        raise ValueError(f"`graph` must be 'knn' or 'radius', got {graph!r}.")


def _moran(
    fa: ad.AnnData, scales: Sequence[float], graph: str, library_key: str | None, n_jobs: int | None
) -> pd.DataFrame:
    scale_col = "k" if graph == "knn" else "radius"
    rows = []
    for scale in scales:
        with _quiet_graph_log():
            _spatial_graph(fa, scale, graph, library_key)
        if fa.obsp["spatial_connectivities"].nnz == 0:  # radius below every neighbor distance: I undefined
            rows.append(pd.DataFrame({"feature": fa.var_names, scale_col: scale, "I": np.nan, "frac_isolated": 1.0}))
            continue
        res = sq.gr.spatial_autocorr(
            fa,
            mode="moran",
            transformation=True,  # row-normalized spatial weights
            n_perms=None,
            copy=True,
            n_jobs=n_jobs,
            show_progress_bar=False,
        )
        df = pd.DataFrame({"feature": res.index, scale_col: scale, "I": res["I"].to_numpy()})
        if graph == "radius":
            # cells without neighbors within the radius have zero spatial lag and shrink I towards 0
            deg = np.diff(fa.obsp["spatial_connectivities"].tocsr().indptr)
            df["frac_isolated"] = float(np.mean(deg == 0))
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def radii_from_knn(
    adata: ad.AnnData,
    ks: Sequence[int] = DEFAULT_KS,
    library_key: str | None = None,
    spatial_key: str = "spatial",
    verbose: bool = True,
) -> np.ndarray:
    """Radii matching kNN neighborhood sizes: median distance to the ``k``-th nearest neighbor (within sections).

    If ``verbose``, the radius used for each ``k`` is printed.
    """
    coords = np.asarray(adata.obsm[spatial_key])
    libs = adata.obs[library_key].astype(str).to_numpy() if library_key is not None else np.zeros(adata.n_obs)
    ks = sorted(int(k) for k in ks)
    dists = []
    for lib in np.unique(libs):
        c = coords[libs == lib]
        d, _ = NearestNeighbors(n_neighbors=min(max(ks) + 1, len(c))).fit(c).kneighbors(c)
        dists.append(d[:, [min(k, d.shape[1] - 1) for k in ks]])
    radii = np.median(np.vstack(dists), axis=0)
    if verbose:
        print("Radii matched to kNN (median distance to the k-th nearest neighbor):")
        for k, r in zip(ks, radii):
            print(f"  k = {k:>4}  ->  radius = {r:.1f}")
    return radii


def moran_profile(
    adata: ad.AnnData,
    ks: Sequence[int] = DEFAULT_KS,
    library_key: str | None = None,
    genes: Sequence[str] | None = None,
    layer: str | None = None,
    obsm_key: str | None = None,
    spatial_key: str = "spatial",
    per_library: bool = False,
    n_jobs: int | None = None,
    graph: Literal["knn", "radius"] = "knn",
    radii: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Moran's I of each feature on spatial kNN (or radius) graphs of increasing size.

    Parameters
    ----------
    adata
        Annotated data with spatial coordinates in ``adata.obsm[spatial_key]``.
    ks
        Neighborhood sizes of the kNN graphs. For ``graph="radius"`` without ``radii``, the radii are matched to
        these neighborhood sizes with :func:`radii_from_knn`.
    library_key
        Column in ``adata.obs`` with the tissue section / FOV. Graphs are built within each section.
    genes
        Genes to analyse (default: all). Ignored if ``obsm_key`` is given.
    layer
        Expression layer to use (default: ``adata.X``).
    obsm_key
        If given, analyse the columns of ``adata.obsm[obsm_key]`` (e.g. a local or global latent) instead of genes.
    spatial_key
        Key of the spatial coordinates in ``adata.obsm``.
    per_library
        If ``False`` (default), one Moran's I is computed on the block-diagonal graph over all sections.
        If ``True``, Moran's I is computed separately in each section.
    n_jobs
        Passed to :func:`squidpy.gr.spatial_autocorr`.
    graph
        ``"knn"`` (default): kNN graphs with ``k`` in ``ks``. ``"radius"``: radius graphs with radii ``radii``.
    radii
        Radii (in units of the spatial coordinates) for ``graph="radius"``. If ``None``, the median distance to the
        ``k``-th nearest neighbor for each ``k`` in ``ks`` is used, so both graph types cover comparable scales.

    Returns
    -------
    Long-format data frame with columns ``feature``, ``k`` (or ``radius`` and ``frac_isolated``), ``I``
    (and ``library`` if ``per_library``).
    """
    if per_library and library_key is None:
        raise ValueError("`per_library=True` requires `library_key`.")
    if graph == "knn":
        if radii is not None:
            raise ValueError("`radii` is only used with `graph='radius'`.")
        scales = sorted(int(k) for k in ks)
    elif graph == "radius":
        if radii is None:
            radii = radii_from_knn(adata, ks=ks, library_key=library_key, spatial_key=spatial_key)
        scales = sorted(float(r) for r in radii)
    else:
        raise ValueError(f"`graph` must be 'knn' or 'radius', got {graph!r}.")
    fa = _feature_adata(adata, genes, layer, obsm_key, library_key, spatial_key)

    if not per_library:
        return _moran(fa, scales, graph, library_key, n_jobs)

    out = []
    for lib in fa.obs[library_key].astype(str).unique():
        sub = fa[fa.obs[library_key].astype(str) == lib].copy()
        if graph == "knn" and sub.n_obs <= max(scales):
            raise ValueError(f"Library {lib!r} has {sub.n_obs} cells, fewer than k={max(scales)} + 1.")
        out.append(_moran(sub, scales, graph, None, n_jobs).assign(library=lib))
    return pd.concat(out, ignore_index=True)


def auc_from_profile(profile: pd.DataFrame, log2_k: bool = False) -> pd.DataFrame:
    """Area under the Moran curve ``I(k)`` per feature (trapezoidal integration over ``k`` or radius).

    Parameters
    ----------
    profile
        Output of :func:`moran_profile`.
    log2_k
        Integrate over ``log2`` of the scale instead of the scale itself, which weights every doubling of the
        neighborhood equally. With the raw scale (default, as in the Methods), the largest interval dominates the area.
        Radius AUCs are in coordinate units and therefore not on the same scale as kNN AUCs; compare ``auc_norm``.

    Returns
    -------
    Data frame with one row per feature (and library, if present) and column ``auc``,
    plus ``auc_norm`` (AUC divided by the integration range, i.e. the mean Moran's I over scales).
    """
    groups = ["feature"] + (["library"] if "library" in profile.columns else [])
    scale_col = "k" if "k" in profile.columns else "radius"

    def _auc(df: pd.DataFrame) -> pd.Series:
        df = df.sort_values(scale_col)
        x = df[scale_col].to_numpy(dtype=float)
        x = np.log2(x) if log2_k else x
        auc = np.trapezoid(df["I"].to_numpy(), x)
        return pd.Series({"auc": auc, "auc_norm": auc / (x[-1] - x[0])})

    return profile.groupby(groups, sort=False).apply(_auc, include_groups=False).reset_index()


def delta_from_profile(profile: pd.DataFrame, split: float) -> pd.DataFrame:
    """Split the decay of the Moran curve into a local (small scales) and a global (large scales) part.

    With scales ``s_min < split < s_max`` (``k`` or radius):

    - ``delta_local = I(s_min) - I(split)``, ``delta_global = I(split) - I(s_max)``
    - ``slope_local`` / ``slope_global``: the deltas divided by the number of doublings of the scale they span
      (``log2(split / s_min)`` and ``log2(s_max / split)``). The two ranges usually span different numbers of scales,
      so the slopes (ΔI per doubling) are the comparable quantities.
    - ``rel_local = delta_local / I(s_min)``, ``rel_global = delta_global / I(split)``: fraction of the
      autocorrelation lost in each range, independent of how strongly autocorrelated the feature is overall.

    Parameters
    ----------
    profile
        Output of :func:`moran_profile` (or a profile averaged over libraries). ``split`` must be one of its scales.
    split
        Scale separating local from global.

    Returns
    -------
    Data frame with one row per feature (and library, if present).
    """
    groups = ["feature"] + (["library"] if "library" in profile.columns else [])
    scale_col = "k" if "k" in profile.columns else "radius"
    scales = np.sort(profile[scale_col].unique())
    if not np.isclose(scales, split).any():
        raise ValueError(f"`split`={split} is not one of the scales {scales.tolist()}.")
    split = scales[np.isclose(scales, split)][0]
    s_min, s_max = scales[0], scales[-1]
    if not s_min < split < s_max:
        raise ValueError(f"`split` must lie strictly between {s_min} and {s_max}.")

    I = profile.pivot_table(index=groups, columns=scale_col, values="I", dropna=False)
    out = pd.DataFrame({"I_min": I[s_min], "I_split": I[split], "I_max": I[s_max]}, index=I.index)
    out["delta_local"] = out["I_min"] - out["I_split"]
    out["delta_global"] = out["I_split"] - out["I_max"]
    out["delta_total"] = out["I_min"] - out["I_max"]
    out["slope_local"] = out["delta_local"] / np.log2(split / s_min)
    out["slope_global"] = out["delta_global"] / np.log2(s_max / split)
    out["rel_local"] = out["delta_local"] / out["I_min"]
    out["rel_global"] = out["delta_global"] / out["I_split"]
    return out.reset_index()


def moran_scale(
    adata: ad.AnnData,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    library_key: str | None = None,
    genes: Sequence[str] | None = None,
    layer: str | None = None,
    obsm_key: str | None = None,
    spatial_key: str = "spatial",
    graph: Literal["knn", "radius"] = "knn",
    radii: Sequence[float] | None = None,
    split: float | None = None,
    per_library: bool = False,
    log2_k: bool = False,
    n_jobs: int | None = None,
    key_added: str = "moran_scale",
    copy: bool = False,
) -> dict[str, pd.DataFrame | dict] | None:
    """Scale-dependent Moran analysis: Moran curves, their AUC and their local / global decay.

    Runs :func:`moran_profile`, :func:`auc_from_profile` and :func:`delta_from_profile` and adds the
    ``local_index = slope_local - slope_global``: positive for features that lose their autocorrelation mainly within
    small neighborhoods (local pattern), negative for features that lose it at tissue scale (global pattern). The local
    index does not depend on how strongly autocorrelated a feature is overall (unlike the AUC).

    Parameters
    ----------
    adata
        Annotated data with spatial coordinates in ``adata.obsm[spatial_key]``.
    ks, library_key, genes, layer, obsm_key, spatial_key, graph, radii, n_jobs
        See :func:`moran_profile`.
    split
        Scale separating local from global (``k`` for ``graph="knn"``, radius for ``graph="radius"``). Must be one of
        the scales. If ``None``, the middle scale (e.g. ``k = 16`` for ``k = 4, ..., 128``).
    per_library
        If ``True``, Moran's I is computed separately in each library and the curves are averaged over libraries
        before the AUC and the deltas are computed. If ``False``, one Moran's I on the graph over all libraries.
    log2_k
        See :func:`auc_from_profile`.
    key_added
        Results are stored in ``adata.uns[key_added]``.
    copy
        If ``True``, return the results instead of storing them.

    Returns
    -------
    If ``copy = True``, a dict with

    - ``profile``: long data frame with ``feature``, ``k`` (or ``radius``) and ``I`` (mean over libraries if
      ``per_library``).
    - ``profile_per_library``: as ``profile``, per library (only if ``per_library``).
    - ``stats``: one row per feature with ``auc``, ``auc_norm``, ``I_min``, ``I_split``, ``I_max``, ``delta_*``,
      ``slope_*``, ``rel_*`` and ``local_index``.
    - ``params``: the parameters (``graph``, ``scale_col``, ``scales``, ``split``, ...).

    Otherwise, stores this dict in ``adata.uns[key_added]`` and returns ``None``.
    """
    profile = moran_profile(
        adata,
        ks=ks,
        library_key=library_key,
        genes=genes,
        layer=layer,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        per_library=per_library,
        n_jobs=n_jobs,
        graph=graph,
        radii=radii,
    )
    scale_col = "k" if graph == "knn" else "radius"
    result: dict[str, pd.DataFrame | dict] = {}
    if per_library:
        result["profile_per_library"] = profile
        # features not expressed in a library are NaN there and skipped in the mean
        value_cols = [c for c in ("I", "frac_isolated") if c in profile.columns]
        profile = profile.groupby(["feature", scale_col], as_index=False, sort=False)[value_cols].mean()

    scales = np.sort(profile[scale_col].unique())
    if split is None:
        split = scales[(len(scales) - 1) // 2]
    stats = delta_from_profile(profile, split=split).merge(auc_from_profile(profile, log2_k=log2_k), on="feature")
    stats["local_index"] = stats["slope_local"] - stats["slope_global"]

    result["profile"] = profile
    result["stats"] = stats
    params = {
        "graph": graph,
        "scale_col": scale_col,
        "scales": scales.tolist(),
        "split": float(scales[np.isclose(scales, split)][0]),
        "layer": layer,
        "obsm_key": obsm_key,
        "library_key": library_key,
        "per_library": per_library,
        "log2_k": log2_k,
    }
    result["params"] = {k: v for k, v in params.items() if v is not None}  # None cannot be written to h5ad
    if copy:
        return result
    adata.uns[key_added] = result
    return None
