import warnings
from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _gene_expression_stats(X, ddof=1):
    """Per-gene detection fraction and standard deviation, without densifying sparse input.

    Returns
    -------
    frac : (G,) fraction of cells with X > 0
    sd : (G,) standard deviation with the given ddof
    """
    n = X.shape[0]

    if sp.issparse(X):
        if X.format not in ("csr", "csc"):
            X = X.tocsr()
        frac = np.asarray((X > 0).sum(axis=0), dtype=float).ravel() / n
        mean = np.asarray(X.mean(axis=0), dtype=float).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0), dtype=float).ravel()
        var = np.maximum(mean_sq - mean**2, 0.0)
        if ddof and n > ddof:
            var = var * (n / (n - ddof))
        sd = np.sqrt(var)
    else:
        Xd = np.asarray(X, dtype=float)
        frac = (Xd > 0).mean(axis=0)
        sd = Xd.std(axis=0, ddof=ddof)

    return frac, sd


def _infer_which(s_key):
    """Infer the component ('local'/'global') from a loading key, or None if ambiguous."""
    if s_key.startswith("_local"):
        return "local"
    if s_key.startswith("_global"):
        return "global"
    return None


def _dim_importance_uns_key(which):
    """Key under which `calculate_dim_importance` stores its selection in adata.uns."""
    return f"_{which}_dim_importance"


def _resolve_dims_from_uns(adata, uns_key, s_key):
    """Read the dimension selection stored by `calculate_dim_importance`.

    Raises with an actionable message if the record is missing, was computed without a
    cumulative cutoff, or came from a different component than `s_key`.
    """
    if uns_key not in adata.uns:
        raise KeyError(
            f"dims=None requires a stored dimension selection, but adata.uns['{uns_key}'] "
            f"is missing. Run calculate_dim_importance(adata, s_key='{s_key}', ...) first, "
            f"or pass dims=[...] explicitly."
        )

    rec = adata.uns[uns_key]

    if "dims_left" not in rec:
        raise KeyError(
            f"adata.uns['{uns_key}'] has no 'dims_left': calculate_dim_importance was run "
            "with use_ratio=False or cumulative_cutoff=None, so no dimensions were selected. "
            "Re-run it with use_ratio=True and a cumulative_cutoff, or pass dims=[...] explicitly."
        )

    # Guard against the mismatch this whole mechanism exists to prevent: a record built
    # from one component being used to pick genes from the other.
    rec_s_key = rec.get("s_key")
    if rec_s_key is not None and str(rec_s_key) != s_key:
        raise ValueError(
            f"Component mismatch: adata.uns['{uns_key}'] was computed from "
            f"s_key='{rec_s_key}', but this call resolved s_key='{s_key}'. "
            "Check the 'which' argument, or re-run calculate_dim_importance for this component."
        )

    # Coming from storage (possibly an h5ad round-trip), so coerce rather than validate
    # strictly -- user-supplied dims keep the strict dtype check below.
    return np.asarray(rec["dims_left"]).ravel().astype(int)


def _get_Z(adata, z_key):
    if z_key not in adata.obsm:
        raise KeyError(f"{z_key} not found in adata.obsm")
    Z = np.asarray(adata.obsm[z_key], dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    ok = np.isfinite(Z).all(axis=1)
    if ok.sum() == 0:
        raise ValueError(f"No finite rows in adata.obsm['{z_key}']")
    return Z[ok]


def latent_rank_report(
    adata,
    *,
    z_key="_global_emb",
    center=True,
    scale=False,
    rtol=1e-7,
    atol=0.0,
):
    """
    Report (approx) linear independence of embedding dimensions using matrix rank.

    Parameters
    ----------
    center : bool
        Subtract column means before rank (recommended).
    scale : bool
        Divide columns by their std before rank (optional).
        Use if dimensions have very different scales.
    rtol, atol : float
        Tolerance parameters passed to np.linalg.matrix_rank.
    """
    Z = _get_Z(adata, z_key)

    # Optional preprocessing
    X = Z.copy()
    if center:
        X -= X.mean(axis=0, keepdims=True)
    if scale:
        sd = X.std(axis=0, ddof=1, keepdims=True)
        # handle constant / near-constant dims robustly
        sd[sd == 0] = 1.0
        X /= sd

    n, d = X.shape
    rank = np.linalg.matrix_rank(X, tol=None)  # uses default SVD-based tol
    # If you want explicit tol control, compute tol yourself; see note below.

    independent = rank == d

    return {
        "z_key": z_key,
        "n_obs": n,
        "n_dims": d,
        "rank": int(rank),
        "linearly_independent": bool(independent),
    }


def get_genes_dim(
    adata,
    which: Literal["global", "local"],  # required
    *,
    dims: Sequence[int] | None = None,  # None -> read the selection stored by calculate_dim_importance
    n_top: int = 20,
    s_key: str | None = None,  # e.g. "_global_std_gene_loadings"
    z_key: str | None = None,  # e.g. "_global_emb" (only used if residualize=True)
    uns_key: str | None = None,  # e.g. "_global_dim_importance"; None -> f"_{which}_dim_importance"
    # expression-based filtering
    X_layer=None,  # e.g. "log1p_norm"; None -> adata.X
    min_frac=0.05,  # fraction of cells with expr>0
    min_sd=None,  # optional sd cutoff on X_layer
    # ranking / scoring
    rank_by="loading",  # "loading" or "loading_x_sd"
    residualize=False,  # downweight correlated dims using z_key
    # optional specificity across provided dims
    enforce_specificity=False,
    specificity_mode="ratio",  # "ratio" or "diff"
    specificity_min=1.5,
    # output ordering
    order_genes_by="winner",  # "winner", "max_abs", or None
    # plotting
    plot=False,
    figsize=(8, 6),
    cmap="BrBG_r",
    title=None,
    show=True,
):
    """
    Compute top genes per dimension and return a DataFrame:
      - index: UNION of selected genes across dims
      - columns: dims
      - values: signed scores (standardized loadings; optionally weighted/residualized)

    If plot=True, also plot a gene×dim heatmap of the returned DataFrame.

    Parameters
    ----------
    which : {"global", "local"}
        Which component to analyse. Required, because it selects the defaults for
        `s_key`, `z_key` and `uns_key` — it is the only argument needed to switch
        components.
    dims : sequence of int, optional
        Latent dimensions to analyse. **Leave as None (recommended)** to read the
        selection `calculate_dim_importance` stored in adata.uns[uns_key], which keeps
        `dims` and `which` from drifting apart. If given explicitly and a stored record
        exists, the two are compared and a mismatch raises a warning.
    uns_key : str, optional
        Record written by `calculate_dim_importance`. Defaults to
        f"_{which}_dim_importance". A record built from a different component than the
        resolved `s_key` raises ValueError rather than silently mixing the two.

    Examples
    --------
    >>> calculate_dim_importance(adata, "global", cumulative_cutoff=0.60)
    >>> df = get_genes_dim(adata, "global", n_top=20)  # dims resolved from adata.uns

    Notes
    -----
    - Uses standardized loadings (S_std) stored in adata.varm[s_key].
    - rank_by="loading_x_sd" multiplies scores by sd(X_g) from X_layer (keeps sign).
      Note this only cancels the per-gene standardization if X_layer is the same layer
      that was passed to `gene_loadings()`.
    - Genes whose score is non-finite in any requested dim are dropped before ranking.
    - enforce_specificity can help pick genes that are strong in one dim and weak in others.
      It compares each gene's largest against its second-largest |score| across the
      *requested* dims only, and is ignored (with a warning) for a single dim.
    - min_frac uses (X > 0), so it is only meaningful for a non-negative layer; on a
      centered/scaled layer roughly half of every gene's values exceed 0.
    - Ties are broken by gene order (stable sort), so the output is reproducible.
    """
    # --- defaults. `which` also names the adata.uns record, so validate it unconditionally.
    if which not in ("local", "global"):
        raise ValueError(f"which must be 'global' or 'local', got {which!r}")

    if s_key is None:
        s_key = f"_{which}_std_gene_loadings"
    if z_key is None:
        z_key = f"_{which}_emb"
    if uns_key is None:
        uns_key = _dim_importance_uns_key(which)

    # --- load/check S
    if s_key not in adata.varm:
        raise KeyError(f"{s_key} not found in adata.varm")
    S = np.asarray(adata.varm[s_key], dtype=float)  # (G, K)
    if S.ndim != 2:
        raise ValueError(f"adata.varm['{s_key}'] must be 2D (genes x dims), got shape {S.shape}")
    genes = np.asarray(adata.var_names)
    K = S.shape[1]

    # --- resolve dims: from adata.uns by default, else cross-check what was passed
    if dims is None:
        dims = _resolve_dims_from_uns(adata, uns_key, s_key)
    elif uns_key in adata.uns and "dims_left" in adata.uns[uns_key]:
        stored = np.asarray(adata.uns[uns_key]["dims_left"]).ravel().astype(int)
        given = np.asarray(dims).ravel()
        # compare as sets: dims_left is importance-ordered, so a reordering is not a mismatch
        if given.dtype != bool and set(given.tolist()) != set(stored.tolist()):
            warnings.warn(
                f"dims={given.tolist()} differs from the selection stored in "
                f"adata.uns['{uns_key}']['dims_left']={stored.tolist()}. Using the dims you "
                "passed. Omit dims to use the stored selection.",
                UserWarning,
                stacklevel=2,
            )

    # --- validate dims (integer indices, in range, unique)
    dims_arr = np.asarray(dims)
    if dims_arr.dtype == bool:
        raise ValueError("dims must be integer dimension indices, not a boolean mask")
    if dims_arr.size and not np.issubdtype(dims_arr.dtype, np.integer):
        raise ValueError(f"dims must be integer dimension indices, got dtype {dims_arr.dtype}")
    dims_arr = dims_arr.ravel()
    if dims_arr.size == 0:
        raise ValueError("dims must be non-empty")
    if np.any((dims_arr < 0) | (dims_arr >= K)):
        raise ValueError(f"dims must be within [0, {K - 1}]")
    if np.unique(dims_arr).size != dims_arr.size:
        raise ValueError("dims must not contain duplicates")
    dims = [int(d) for d in dims_arr]

    # --- validate remaining scalar arguments up front
    if rank_by not in ("loading", "loading_x_sd"):
        raise ValueError("rank_by must be 'loading' or 'loading_x_sd'")
    if order_genes_by not in ("winner", "max_abs", None):
        raise ValueError("order_genes_by must be 'winner', 'max_abs', or None")
    n_top = int(n_top)
    if n_top < 1:
        raise ValueError(f"n_top must be >= 1, got {n_top}")

    score_full = S.copy()

    # --- optional residualization (uniqueness factor per dim)
    if residualize:
        if z_key not in adata.obsm:
            raise KeyError(f"{z_key} not found in adata.obsm (needed for residualize=True)")
        Z = np.asarray(adata.obsm[z_key], dtype=float)
        if Z.ndim == 1:
            Z = Z[:, None]
        if Z.shape[1] != K:
            raise ValueError(f"Latent dim mismatch: S has {K}, Z has {Z.shape[1]}")

        ok = np.isfinite(Z).all(axis=1)
        if ok.sum() < 3:
            raise ValueError(f"Need >=3 finite rows in adata.obsm['{z_key}']")
        Z = Z[ok]
        Zc = Z - Z.mean(axis=0, keepdims=True)

        sd = Zc.std(axis=0, ddof=1)
        sd[sd == 0] = np.nan

        uniq = np.ones(K, dtype=float)
        for k in range(K):
            y = Zc[:, k]
            Xr = np.delete(Zc, k, axis=1)
            if Xr.shape[1] == 0:
                r = y
            else:
                b, *_ = np.linalg.lstsq(Xr, y, rcond=None)
                r = y - Xr @ b
            rsd = np.std(r, ddof=1)
            uniq[k] = (rsd / sd[k]) if np.isfinite(rsd) and np.isfinite(sd[k]) and sd[k] > 0 else 0.0

        score_full = score_full * uniq[None, :]

    # --- expression statistics (only if a filter or the sd weighting needs them)
    need_frac = min_frac is not None and min_frac > 0
    need_sd = (min_sd is not None) or (rank_by == "loading_x_sd")

    sd_g = None
    if need_frac or need_sd:
        X = adata.layers[X_layer] if X_layer is not None else adata.X
        frac, sd_g = _gene_expression_stats(X)  # sparse-aware, no densification

    # --- apply sd weighting to scores if requested (keeps sign)
    score_used = score_full
    if rank_by == "loading_x_sd":
        score_used = score_used * sd_g[:, None]

    # --- gene filters. Non-finite scores must be excluded: np.argsort sends NaN to the
    # end of the array, so a descending sort would otherwise rank NaN genes first.
    mask = np.isfinite(score_used[:, dims]).all(axis=1)
    if sd_g is not None:
        mask &= np.isfinite(sd_g)
        if need_frac:
            mask &= frac >= min_frac
        if min_sd is not None:
            mask &= sd_g >= min_sd

    valid = np.where(mask)[0]
    if valid.size == 0:
        raise ValueError("No genes passed filters. Relax min_frac/min_sd or check X_layer.")

    # --- optional specificity filter across chosen dims
    valid2 = valid
    if enforce_specificity:
        if specificity_mode not in ("ratio", "diff"):
            raise ValueError("specificity_mode must be 'ratio' or 'diff'")

        if len(dims) < 2:
            warnings.warn(
                "enforce_specificity=True has no effect with a single dim; ignoring it.",
                UserWarning,
                stacklevel=2,
            )
        else:
            A = np.abs(score_used[np.ix_(valid, dims)])  # (n_valid, n_dims)
            maxv = A.max(axis=1)
            second = np.partition(A, -2, axis=1)[:, -2]
            eps = 1e-12

            if specificity_mode == "ratio":
                keep = (maxv / (second + eps)) >= specificity_min
            else:
                keep = (maxv - second) >= specificity_min

            valid2 = valid[keep]
            if valid2.size == 0:
                raise ValueError("No genes passed specificity filter. Lower specificity_min or disable it.")

    # --- union of top genes per dim based on |score|
    selected = set()
    for d in dims:
        sc = np.abs(score_used[valid2, d])
        top_idx = valid2[np.argsort(-sc, kind="stable")[:n_top]]
        selected.update(top_idx.tolist())

    selected = np.array(sorted(selected), dtype=int)
    sel_genes = genes[selected]

    mat = score_used[np.ix_(selected, dims)]
    df = pd.DataFrame(mat, index=sel_genes, columns=[str(d) for d in dims])

    # --- optional ordering of genes
    if order_genes_by == "winner" and df.shape[1] >= 1:
        A = np.abs(df.values)
        winner = np.argmax(A, axis=1)
        strength = A[np.arange(A.shape[0]), winner]
        # primary key: winning dim; secondary: descending |score| within that dim, so
        # each winner block reads strongest-first (np.lexsort takes the last key first)
        df = df.iloc[np.lexsort((-strength, winner))]
    elif order_genes_by == "max_abs":
        mx = np.max(np.abs(df.values), axis=1)
        df = df.iloc[np.argsort(-mx, kind="stable")]

    # --- optional plot
    ax = None
    if plot:
        vmax = np.nanmax(np.abs(df.values))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(df.values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046)

        ax.set_xticks(np.arange(df.shape[1]))
        ax.set_xticklabels(df.columns.tolist())

        ax.set_yticks(np.arange(df.shape[0]))
        ax.set_yticklabels(df.index.tolist())

        ax.set_xlabel("latent dimension")
        ax.set_ylabel("gene")

        # ---- draw separators between winner blocks
        # if order_genes_by == "winner" and df.shape[1] > 1:
        #    winner = np.argmax(np.abs(df.values), axis=1)
        #    change_points = np.where(np.diff(winner) != 0)[0]

        #    for cp in change_points:
        #        ax.axhline(cp + 0.5, color="white", linewidth=4)

        ax.set_title(title or f"Gene×dim loadings (which={which}, top={n_top}/dim, residualize={residualize})")

        plt.tight_layout()
        if show:
            plt.show()

    return df if not plot else (df, ax)


def calculate_dim_importance(
    adata,
    which: Literal["global", "local"] | None = None,
    *,
    s_key: str | None = None,
    z_key: str | None = None,
    mode: Literal["full", "diag"] = "full",
    use_ratio: bool = True,
    cumulative_cutoff: float | None = 0.90,
    spacing: int = 2,
    n_top: int | None = None,
    uns_key: str | None = None,
    store: bool = True,
):
    """Calculate dimension importance scores.

    Also stores the selected dimensions in ``adata.uns[uns_key]`` so that
    `get_genes_dim` can pick them up automatically (see `store`).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    which : {"global", "local"}, optional
        Which component to score. Derives `s_key` and `z_key`, so it is normally the
        only argument needed. Defaults to "global" when `s_key` is not given; when
        `s_key` is given instead, the component is inferred from it. Passing a `which`
        that contradicts `s_key` is an error.
    s_key : str, optional
        Key in adata.varm containing gene loadings. Defaults to
        f"_{which}_std_gene_loadings"; only needed for non-standard keys.
    z_key : str, optional
        Key in adata.obsm containing embedding. Defaults to f"_{which}_emb".
    mode : str
        "full" (uses Corr(Z) off-diagonals) or "diag" (assumes dims uncorrelated).
    use_ratio : bool
        If True, normalize to sum to 1.
    cumulative_cutoff : float
        Threshold for cumulative variance.
    spacing : int
        Spacing between points on x-axis.
    n_top : int, optional
        Maximum number of dimensions to include.
    uns_key : str, optional
        Where to store the result. Defaults to f"_{which}_dim_importance".
    store : bool
        If True (default), write the selection to adata.uns[uns_key] so that
        `get_genes_dim(adata, which=which)` can resolve `dims` without being told.

    Returns
    -------
    dict
        Dictionary containing:
        - y_plot : importance scores (subset)
        - dim_plot : dimension indices (subset)
        - y_sorted : all sorted scores
        - dim_sorted : all sorted indices
        - cutoff_x : x-coordinate of cutoff line (or None)
        - cutoff_idx : index of cutoff (or None)
        - n_dims_left : number of dims at cutoff (or None)
        - dims_left : dimension indices at cutoff (or None)
        - cumulative_cutoff : threshold used
        - mode : mode used
        - s_key : loadings key used
        - z_key : embedding key used
        - which : component inferred or given ("local"/"global", or None)
        - uns_key : where the selection was stored (or None if not stored)
    """
    # --- resolve component and keys. `which` is the normal entry point; s_key/z_key
    # override it for non-standard keys. Never let the two disagree silently: that would
    # file a local result under the global record (the mix-up this record exists to stop).
    if which is not None and which not in ("local", "global"):
        raise ValueError(f"which must be 'global' or 'local', got {which!r}")

    inferred = _infer_which(s_key) if s_key is not None else None

    if which is None:
        # infer from an explicit s_key, else keep the historical "global" default
        which = inferred if inferred is not None else ("global" if s_key is None else None)
    elif inferred is not None and inferred != which:
        raise ValueError(
            f"which={which!r} contradicts s_key={s_key!r}, which looks like '{inferred}'. Pass only one of the two."
        )

    if which is None and (s_key is None or z_key is None):
        raise ValueError(
            f"Cannot derive the missing keys from s_key={s_key!r}. Pass which='local'/'global', "
            "or give both s_key and z_key explicitly."
        )

    if s_key is None:
        s_key = f"_{which}_std_gene_loadings"
    if z_key is None:
        z_key = f"_{which}_emb"

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
    else:
        y = score

    # -------------------
    # sorting
    # -------------------

    order = np.argsort(y)[::-1]
    y_sorted = y[order]
    dim_sorted = order

    # -------------------
    # cutoff calculation
    # -------------------

    cutoff_x = None
    cutoff_idx = None
    n_dims_left = None
    dims_left = None

    if use_ratio and cumulative_cutoff is not None and y.sum() > 0:
        cum = np.cumsum(y_sorted)
        cutoff_idx = int(np.searchsorted(cum, cumulative_cutoff, side="left"))
        cutoff_x = cutoff_idx * spacing

        n_dims_left = cutoff_idx + 1
        dims_left = dim_sorted[:n_dims_left]

    # -------------------
    # apply n_top filter
    # -------------------

    K = len(y_sorted)

    if n_top is not None:
        K = min(n_top, K)

    y_plot = y_sorted[:K]
    dim_plot = dim_sorted[:K]

    # -------------------
    # persist the selection so get_genes_dim can resolve `dims` on its own
    # -------------------

    if uns_key is None and which is not None:
        uns_key = _dim_importance_uns_key(which)

    if store:
        if uns_key is None:
            warnings.warn(
                f"Could not infer the component from s_key='{s_key}', so the dimension "
                "selection was not stored. Pass which='local'/'global' or uns_key=... to "
                "enable get_genes_dim(adata, ...) to resolve dims automatically.",
                UserWarning,
                stacklevel=2,
            )
        else:
            # anndata cannot write None into .uns, so omit absent entries rather than
            # storing None -- dims_left is None whenever no cutoff was applied.
            record = {
                "dims_sorted": np.asarray(dim_sorted, dtype=int),
                "importance_sorted": np.asarray(y_sorted, dtype=float),
                "n_dims": int(len(y_sorted)),
                "mode": str(mode),
                "s_key": str(s_key),
                "z_key": str(z_key),
                "use_ratio": bool(use_ratio),
            }
            if which is not None:
                record["which"] = str(which)
            if dims_left is not None:
                record["dims_left"] = np.asarray(dims_left, dtype=int)
                record["n_dims_left"] = int(n_dims_left)
            if cumulative_cutoff is not None:
                record["cumulative_cutoff"] = float(cumulative_cutoff)

            adata.uns[uns_key] = record

    return {
        "y_plot": y_plot,
        "dim_plot": dim_plot,
        "y_sorted": y_sorted,
        "dim_sorted": dim_sorted,
        "cutoff_x": cutoff_x,
        "cutoff_idx": cutoff_idx,
        "n_dims_left": n_dims_left,
        "dims_left": dims_left,
        "cumulative_cutoff": cumulative_cutoff,
        "mode": mode,
        "s_key": s_key,
        "z_key": z_key,
        "use_ratio": use_ratio,
        "spacing": spacing,
        "which": which,
        "uns_key": uns_key,
    }
