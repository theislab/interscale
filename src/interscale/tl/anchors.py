"""Anchor points of a latent dimension, and expression profiles at range from them.

Motivation
----------
A *global* InterScale dimension is free to draw on the whole sample, so a claim that it
carries a long-range interaction needs the interaction shown at a distance. These functions
localise the tissue that drives a dimension (its **anchor foci**), measure every other cell's
distance to those foci, and profile expression against that distance:

    anchor, cluster = find_anchor_cells(adata, "43_global_emb", 13)   # a handful of foci
    d = anchor_signed_distance(adata, anchor)                         # <0 inside, >0 outside
    zone = anchor_zones(adata, d, near_max=local_reach_um(cfg))       # core / near / far
    prof = profile_by_distance(adata, ["PTCH1", "NKX6.1"], d)         # binned mean +/- SEM

The construction mirrors the fibrosis-core panels of the multi-organ supplement: a
thresholded score defines cores, cores get a signed distance to their border, and covariates
are plotted against that signed distance. The difference is that the core here is not an
annotation -- it is read off the model.

Why "signed distance to the border" and not "distance to the nearest anchor cell": the latter
is zero for every cell of a focus regardless of how deep inside it sits, which collapses the
whole core into one bin. Negative depth keeps the core resolvable, and 0 is the border in both
directions, so a profile is continuous across it.

Why the distance axis is the point
----------------------------------
The local component sees `num_layers` hops of a radius-`radius` graph, i.e. at most
`num_layers * radius` micrometers (:func:`local_reach_um`). Structure in a profile *beyond*
that abscissa cannot have come from the local component, which is precisely what makes it
evidence of a long-range interaction rather than a neighbourhood effect. Every plot in
``interscale.pl.anchor_plots`` therefore marks that reach.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

__all__ = [
    "local_reach_um",
    "find_anchor_cells",
    "anchor_signed_distance",
    "anchor_zones",
    "profile_by_distance",
    "anchor_enrichment",
]

# Zone names. Module constants because the plotting side has to order and colour them, and a
# typo in one place would silently produce an unordered categorical.
ZONE_CORE = "anchor core"
ZONE_NEAR = "near"
ZONE_FAR = "far"
ZONE_ORDER = (ZONE_CORE, ZONE_NEAR, ZONE_FAR)


def local_reach_um(cfg) -> float:
    """Furthest a cell's own representation can travel through the local component, in um.

    ``num_layers`` message-passing hops over a radius-``radius`` graph, so ``num_layers *
    radius`` is a hard bound -- the *typical* walk is shorter (see
    :func:`interscale.tl.get_average_local_and_global_size`, which measures it from the
    realised edge lengths). The bound is the useful number here: anything past it is out of
    the local component's reach for every cell, not just the average one.
    """
    radius = cfg.dataset.spatial_neigbors_kwargs.radius
    if radius is None:
        raise ValueError(
            "cfg.dataset.spatial_neigbors_kwargs.radius is None, so the local component's "
            "reach is a neighbour count rather than a distance and cannot be expressed in um."
        )
    return float(radius) * int(cfg.model.local_component.parameters.num_layers)


def _grouping(adata, sample_key, samples):
    """(group labels, groups to visit). Restricting to `samples` is not cosmetic: the
    threshold is per group, so visiting a group means thresholding inside it.
    """
    grp = adata.obs[sample_key].astype(str).to_numpy()
    if samples is None:
        return grp, np.unique(grp)
    todo = np.asarray([str(s) for s in np.atleast_1d(samples)])
    missing = set(todo) - set(np.unique(grp))
    if missing:
        raise KeyError(f"adata.obs['{sample_key}'] has no group(s) {sorted(missing)}")
    return grp, todo


def find_anchor_cells(
    adata,
    emb_key: str,
    dim: int,
    *,
    sample_key: str = "sample",
    spatial_key: str = "spatial",
    quantile: float = 0.95,
    sign: str = "auto",
    link_radius: float = 400.0,
    min_cluster_size: int = 8,
    samples: Sequence[str] | None = None,
    key_added: str | None = None,
):
    """Cells driving one latent dimension, kept only where they form a spatial focus.

    Two filters, in this order:

    1. **Tail of the dimension**, thresholded at ``quantile`` *within each sample*. Per
       sample, not globally: the samples are separate graphs with their own global embedding,
       and one sample's offset would otherwise take every anchor in the dataset.
    2. **Spatial coherence** -- single-linkage clustering of the tail cells at
       ``link_radius``, discarding components smaller than ``min_cluster_size``. This is what
       turns "the top 5% of cells" into "a handful of anchor points": scattered high-score
       cells carry no location to measure a distance from, and dropping them is the difference
       between a few foci and hundreds of singletons.

    Parameters
    ----------
    emb_key
        ``adata.obsm`` key, e.g. ``f"{seed}_global_emb"``.
    dim
        Column of that embedding. Normally one of the dimensions
        :func:`interscale.evaluation.calculate_dim_importance` kept.
    quantile
        Per-sample score quantile the tail starts at. 0.95 is a deliberate compromise: at 0.99
        a ~1800-cell sample offers only ~18 tail cells, too few to survive
        ``min_cluster_size`` unless they sit on top of each other, so most dimensions come
        back with no focus at all.
    sign
        ``"+"`` / ``"-"`` to take the high or low tail; ``"auto"`` (default) takes whichever
        tail reaches further from the median. A dimension's sign is arbitrary -- it flips with
        the decoder weight column -- so hardcoding ``"+"`` would silently analyse the wrong
        end for half the dimensions.
    link_radius
        Single-linkage distance for grouping tail cells into foci. Default 400 um =
        :func:`local_reach_um` for the legnini23 configs (2 hops x 200 um), i.e. cells close
        enough that the local component could have linked them count as one focus.
    min_cluster_size
        Foci smaller than this are dropped entirely (not marked as anchors).
    samples
        Restrict to these ``sample_key`` groups. Everything else gets ``anchor=False`` and
        ``cluster=-1``; the thresholds are unaffected, being per sample already.
    key_added
        If given, write ``adata.obs[key_added]`` (bool) and
        ``adata.obs[f"{key_added}_focus"]`` (int, -1 = not an anchor).

    Returns
    -------
    anchor : (n_obs,) bool
    cluster : (n_obs,) int
        Focus id, unique across samples; ``-1`` where ``anchor`` is False.

    Notes
    -----
    ``adata.uns[f"{key_added}_info"]`` records the dimension, the sign taken and the
    per-sample thresholds, so a figure can state what it thresholded rather than the caller
    having to remember.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    if sign not in ("auto", "+", "-"):
        raise ValueError(f"sign must be 'auto', '+' or '-', got {sign!r}")

    if emb_key not in adata.obsm:
        raise KeyError(f"{emb_key} not found in adata.obsm")
    Z = np.asarray(adata.obsm[emb_key], dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    dim = int(dim)
    if not -Z.shape[1] <= dim < Z.shape[1]:
        raise ValueError(f"dim {dim} out of range for {emb_key} with {Z.shape[1]} dimensions")

    xy = np.asarray(adata.obsm[spatial_key], dtype=float)
    grp, todo = _grouping(adata, sample_key, samples)

    s = Z[:, dim]
    if sign == "auto":
        med = np.nanmedian(s)
        lo, hi = np.nanquantile(s, [1.0 - quantile, quantile])
        sgn = 1.0 if abs(hi - med) >= abs(med - lo) else -1.0
    else:
        sgn = 1.0 if sign == "+" else -1.0
    score = sgn * s

    anchor = np.zeros(adata.n_obs, dtype=bool)
    cluster = np.full(adata.n_obs, -1, dtype=int)
    thresholds, next_id = {}, 0

    for g in todo:
        rows = np.where(grp == g)[0]
        thr = float(np.nanquantile(score[rows], quantile))
        thresholds[str(g)] = thr
        cand = rows[score[rows] >= thr]
        if cand.size == 0:
            continue

        # Single linkage via connected components of the <=link_radius graph. Built as a
        # sparse distance matrix so this stays linear-ish in the number of close pairs
        # instead of materialising cand x cand.
        tree = cKDTree(xy[cand])
        pairs = tree.sparse_distance_matrix(tree, link_radius, output_type="coo_matrix")
        n_comp, labels = connected_components(
            csr_matrix((np.ones_like(pairs.data, dtype=bool), (pairs.row, pairs.col)), shape=(cand.size, cand.size)),
            directed=False,
        )
        for c in range(n_comp):
            idx = cand[labels == c]
            if idx.size >= min_cluster_size:
                anchor[idx] = True
                cluster[idx] = next_id
                next_id += 1

    if key_added is not None:
        adata.obs[key_added] = anchor
        adata.obs[f"{key_added}_focus"] = cluster
        adata.uns[f"{key_added}_info"] = {
            "emb_key": str(emb_key),
            "dim": dim,
            "sign": sgn,
            "quantile": float(quantile),
            "link_radius": float(link_radius),
            "min_cluster_size": int(min_cluster_size),
            "n_foci": int(next_id),
            "thresholds": thresholds,
        }
    return anchor, cluster


def anchor_signed_distance(
    adata,
    anchor,
    *,
    sample_key: str = "sample",
    spatial_key: str = "spatial",
    samples: Sequence[str] | None = None,
    key_added: str | None = None,
):
    """Signed distance to the anchor border: negative inside a focus, positive outside.

    Per sample, and per sample only -- distances are computed within a ``sample_key`` group
    and never across groups, so a cell is never measured against an anchor on another slide.

    A cell outside every focus gets ``+`` its distance to the nearest anchor cell; a cell
    inside one gets ``-`` its distance to the nearest non-anchor cell, i.e. how deep in the
    focus it sits. Both are zero at the border, so the two branches join up.

    Returns
    -------
    (n_obs,) float, ``nan`` for samples that were skipped or that hold no anchor (a Ctrl
    sample where the dimension found nothing is a *result*, so it is not an error) and for
    samples that are entirely anchor (no border to measure against).
    """
    anchor = np.asarray(anchor, dtype=bool)
    if anchor.shape != (adata.n_obs,):
        raise ValueError(f"anchor must have shape ({adata.n_obs},), got {anchor.shape}")

    xy = np.asarray(adata.obsm[spatial_key], dtype=float)
    grp, todo = _grouping(adata, sample_key, samples)

    out = np.full(adata.n_obs, np.nan)
    for g in todo:
        rows = np.where(grp == g)[0]
        inside, outside = rows[anchor[rows]], rows[~anchor[rows]]
        if inside.size == 0 or outside.size == 0:
            continue
        out[outside] = cKDTree(xy[inside]).query(xy[outside])[0]
        out[inside] = -cKDTree(xy[outside]).query(xy[inside])[0]

    if key_added is not None:
        adata.obs[key_added] = out
    return out


def anchor_zones(adata, dist, *, near_max: float = 400.0, key_added: str | None = None):
    """Split a signed distance into ``anchor core`` / ``near`` / ``far``.

    ``near_max`` is the local component's reach (:func:`local_reach_um`), which makes the cut
    mean something: ``near`` is the band a local model could have explained on its own, and
    ``far`` is the band only the global component can reach. Returns an ordered categorical
    (``nan`` distances stay unassigned) so plots and groupbys keep core -> near -> far order.
    """
    dist = np.asarray(dist, dtype=float)
    zone = np.full(adata.n_obs, None, dtype=object)
    ok = np.isfinite(dist)
    zone[ok & (dist <= 0)] = ZONE_CORE
    zone[ok & (dist > 0) & (dist <= near_max)] = ZONE_NEAR
    zone[ok & (dist > near_max)] = ZONE_FAR
    out = pd.Categorical(zone, categories=list(ZONE_ORDER), ordered=True)
    if key_added is not None:
        adata.obs[key_added] = out
    return out


def profile_by_distance(
    adata,
    genes: Sequence[str],
    dist,
    *,
    layer: str | None = "log1p_norm",
    mask=None,
    n_bins: int = 14,
    min_cells: int = 15,
    binning: str = "quantile",
):
    """Mean expression per distance bin, one row per (gene, bin).

    Parameters
    ----------
    layer
        ``adata.layers`` key holding the expression to profile, or ``None`` for ``.X``. Use
        the layer the decoders were trained on, so the profile is on the same scale as the
        loadings that selected `genes`.
    mask
        Boolean over ``adata.n_obs``, normally one slide. Combined with ``isfinite(dist)``.
    binning
        ``"quantile"`` (default) puts a comparable number of cells in each bin, which keeps
        the error bars comparable along the axis; ``"equal"`` uses equal-width bins, which is
        the honest choice if the *shape* against distance is being read off rather than
        compared bin to bin. Bins under ``min_cells`` are dropped either way.

    Returns
    -------
    DataFrame with ``gene, bin, lo, hi, center, n, mean, sem``. ``center`` is the mean
    distance of the cells in the bin, not the bin midpoint, so a point sits where its cells
    actually are.
    """
    if binning not in ("quantile", "equal"):
        raise ValueError(f"binning must be 'quantile' or 'equal', got {binning!r}")

    genes = list(genes)
    missing = [g for g in genes if g not in adata.var_names]
    if missing:
        raise KeyError(f"genes not in adata.var_names: {missing}")

    dist = np.asarray(dist, dtype=float)
    ok = np.isfinite(dist)
    if mask is not None:
        ok &= np.asarray(mask, dtype=bool)
    if ok.sum() == 0:
        return pd.DataFrame(columns=["gene", "bin", "lo", "hi", "center", "n", "mean", "sem"])

    sub = adata[ok, genes]
    X = sub.layers[layer] if layer is not None else sub.X
    X = np.asarray(X.todense() if issparse(X) else X, dtype=float)
    d = dist[ok]

    if binning == "quantile":
        edges = np.unique(np.quantile(d, np.linspace(0.0, 1.0, n_bins + 1)))
    else:
        edges = np.linspace(d.min(), d.max(), n_bins + 1)
    if edges.size < 2:
        return pd.DataFrame(columns=["gene", "bin", "lo", "hi", "center", "n", "mean", "sem"])
    # digitize on the interior edges only, then clip, so the extremes land in the end bins
    # rather than in out-of-range bin 0 / bin n+1.
    which = np.clip(np.digitize(d, edges[1:-1], right=True), 0, edges.size - 2)

    rows = []
    for b in range(edges.size - 1):
        sel = which == b
        n = int(sel.sum())
        if n < min_cells:
            continue
        vals = X[sel]
        # ddof=1 needs n >= 2; min_cells is well above that, but sem is meaningless at n=1.
        sem = vals.std(axis=0, ddof=1) / np.sqrt(n)
        for j, gene in enumerate(genes):
            rows.append(
                {
                    "gene": gene,
                    "bin": b,
                    "lo": float(edges[b]),
                    "hi": float(edges[b + 1]),
                    "center": float(d[sel].mean()),
                    "n": n,
                    "mean": float(vals[:, j].mean()),
                    "sem": float(sem[j]),
                }
            )
    return pd.DataFrame(rows)


def anchor_enrichment(adata, anchor, obs_key, *, sample_key="sample", samples=None):
    """What the anchor foci are made of, per sample: composition inside vs in the rest.

    Answers "did this dimension anchor on anything nameable?" -- e.g. whether a dimension's
    foci coincide with the SHH-source graft. ``log2_enrichment`` is inside-vs-rest on the
    fraction of cells; it is ``inf`` for a category absent outside the foci and ``-inf`` for
    one absent inside, which is the honest reading of a category that is exclusive to one
    side.

    Returns a DataFrame with ``sample, category, n_in, n_out, frac_in, frac_out,
    log2_enrichment``, empty for samples with no anchor.
    """
    anchor = np.asarray(anchor, dtype=bool)
    grp, todo = _grouping(adata, sample_key, samples)
    labels = adata.obs[obs_key].astype(str).to_numpy()
    cats = pd.unique(adata.obs[obs_key].astype(str))

    rows = []
    for g in todo:
        m = grp == g
        inside, outside = m & anchor, m & ~anchor
        if inside.sum() == 0:
            continue
        for c in cats:
            n_in = int((labels[inside] == c).sum())
            n_out = int((labels[outside] == c).sum())
            f_in = n_in / inside.sum()
            f_out = n_out / outside.sum() if outside.sum() else np.nan
            with np.errstate(divide="ignore", invalid="ignore"):
                lfc = np.log2(f_in / f_out) if f_in or f_out else np.nan
            rows.append(
                {
                    "sample": str(g),
                    "category": c,
                    "n_in": n_in,
                    "n_out": n_out,
                    "frac_in": f_in,
                    "frac_out": f_out,
                    "log2_enrichment": float(lfc),
                }
            )
    return pd.DataFrame(rows)
