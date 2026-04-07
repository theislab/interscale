import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

    independent = (rank == d)

    return {
        "z_key": z_key,
        "n_obs": n,
        "n_dims": d,
        "rank": int(rank),
        "linearly_independent": bool(independent),
    }

def pl_latent_correlation(
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
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(),
        rotation=90,
        fontsize=label_fontsize
    )

    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(),
        rotation=0,
        fontsize=label_fontsize
    )

    g.ax_heatmap.set_xlabel("latent dim")
    g.ax_heatmap.set_ylabel("latent dim")
    #g.ax_heatmap.set_title(title or f"Correlation of {z_key}", pad=12)

    plt.tight_layout()

    if show:
        plt.show()

    return g

def pl_dim_importance_elbow_stdexpr(
    adata,
    s_key="_global_std_gene_loadings",
    z_key="_global_emb",
    mode="full",
    use_ratio=True,
    cumulative_cutoff=0.90,
    spacing=2,
    n_top=None,                 # NEW
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

        print(
            f"\n{int(cumulative_cutoff*100)}% cumulative variance reached with "
            f"{n_dims_left} dimensions:"
        )
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

def get_genes_dim(
    adata,
    dims,
    which="global",                    # "global" or "local"
    n_top=20,
    s_key=None,                        # e.g. "_global_std_gene_loadings"
    z_key=None,                        # e.g. "_global_emb" (only used if residualize=True)
    # expression-based filtering 
    X_layer=None,                      # e.g. "log1p_norm"; None -> adata.X
    min_frac=0.05,                     # fraction of cells with expr>0
    min_sd=None,                       # optional sd cutoff on X_layer
    # ranking / scoring
    rank_by="loading",                 # "loading" or "loading_x_sd"
    residualize=False,                 # downweight correlated dims using z_key
    # optional specificity across provided dims
    enforce_specificity=False,
    specificity_mode="ratio",          # "ratio" or "diff"
    specificity_min=1.5,
    # output ordering
    order_genes_by="winner",           # "winner", "max_abs", or None
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

    Notes
    -----
    - Uses standardized loadings (S_std) stored in adata.varm[s_key].
    - rank_by="loading_x_sd" multiplies scores by sd(X_g) from X_layer (keeps sign).
    - enforce_specificity can help pick genes that are strong in one dim and weak in others.
    """

    # --- defaults
    if s_key is None:
        if which == "global":
            s_key = "_global_std_gene_loadings"
        elif which == "local":
            s_key = "_local_std_gene_loadings"
        else:
            raise ValueError("which must be 'global' or 'local' (or provide s_key explicitly)")

    if z_key is None:
        if which == "global":
            z_key = "_global_emb"
        elif which == "local":
            z_key = "_local_emb"

    # --- load/check S
    if s_key not in adata.varm:
        raise KeyError(f"{s_key} not found in adata.varm")
    S = np.asarray(adata.varm[s_key], dtype=float)  # (G, K)
    genes = np.asarray(adata.var_names)
    G, K = S.shape

    dims = list(dims)
    if len(dims) == 0:
        raise ValueError("dims must be non-empty")
    if any((d < 0 or d >= K) for d in dims):
        raise ValueError(f"dims must be within [0, {K-1}]")

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

    # --- expression-based filters and optional sd weighting
    X = adata.layers[X_layer] if X_layer is not None else adata.X
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=float)  # (N, G)

    frac = (X > 0).mean(axis=0)
    sd_g = X.std(axis=0, ddof=1)

    mask = np.isfinite(sd_g) & (frac >= min_frac)
    if min_sd is not None:
        mask &= (sd_g >= min_sd)

    valid = np.where(mask)[0]
    if valid.size == 0:
        raise ValueError("No genes passed filters. Relax min_frac/min_sd or check X_layer.")

    # --- apply sd weighting to scores if requested (keeps sign)
    score_used = score_full
    if rank_by == "loading_x_sd":
        score_used = score_used * sd_g[:, None]
    elif rank_by != "loading":
        raise ValueError("rank_by must be 'loading' or 'loading_x_sd'")

    # --- optional specificity filter across chosen dims
    valid2 = valid
    if enforce_specificity and len(dims) >= 2:
        A = np.abs(score_used[np.ix_(valid, dims)])  # (n_valid, n_dims)
        maxv = A.max(axis=1)
        second = np.partition(A, -2, axis=1)[:, -2]
        eps = 1e-12

        if specificity_mode == "ratio":
            keep = (maxv / (second + eps)) >= specificity_min
        elif specificity_mode == "diff":
            keep = (maxv - second) >= specificity_min
        else:
            raise ValueError("specificity_mode must be 'ratio' or 'diff'")

        valid2 = valid[keep]
        if valid2.size == 0:
            raise ValueError("No genes passed specificity filter. Lower specificity_min or disable it.")

    # --- union of top genes per dim based on |score|
    selected = set()
    for d in dims:
        sc = np.abs(score_used[valid2, d])
        top_idx = valid2[np.argsort(sc)[::-1][:n_top]]
        selected.update(top_idx.tolist())

    selected = np.array(sorted(selected), dtype=int)
    sel_genes = genes[selected]

    mat = score_used[np.ix_(selected, dims)]
    df = pd.DataFrame(mat, index=sel_genes, columns=[str(d) for d in dims])

    # --- optional ordering of genes
    if order_genes_by == "winner" and df.shape[1] >= 1:
        winner = np.argmax(np.abs(df.values), axis=1)
        df = df.iloc[np.argsort(winner)]
    elif order_genes_by == "max_abs":
        mx = np.max(np.abs(df.values), axis=1)
        df = df.iloc[np.argsort(-mx)]
    elif order_genes_by is None:
        pass
    else:
        raise ValueError("order_genes_by must be 'winner', 'max_abs', or None")

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
        #if order_genes_by == "winner" and df.shape[1] > 1:
        #    winner = np.argmax(np.abs(df.values), axis=1)
        #    change_points = np.where(np.diff(winner) != 0)[0]

        #    for cp in change_points:
        #        ax.axhline(cp + 0.5, color="white", linewidth=4)

        ax.set_title(
            title
            or f"Gene×dim loadings (which={which}, top={n_top}/dim, residualize={residualize})"
        )

        plt.tight_layout()
        if show:
            plt.show()

    return df if not plot else (df, ax)