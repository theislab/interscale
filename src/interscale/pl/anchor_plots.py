"""Figures for the anchor / distance-zone analysis of :mod:`interscale.tl.anchors`.

The panels here are the visual argument that a *global* InterScale dimension carries a
long-range interaction rather than a neighbourhood effect:

    anchor, focus = find_anchor_cells(adata, "43_global_emb", 13, key_added="anchor13")
    dist  = anchor_signed_distance(adata, anchor)
    zone  = anchor_zones(adata, dist, near_max=local_reach_um(cfg))

    anchor_map(adata, anchor, samples=slides)            # where the dimension anchors
    signed_distance_map(adata, dist, samples=slides)     # distance to the anchor border
    zone_map(adata, zone, samples=slides)                # core / near / far
    gene_maps(adata, genes, samples=slides)              # the dimension's genes, per slide
    distance_profile(prof, local_reach=local_reach_um(cfg))   # those genes vs distance

Every distance plot marks ``local_reach``, the furthest a cell's own representation can
travel through the local component (:func:`interscale.tl.local_reach_um`). That line is the
point of the figure: structure to the right of it is out of the local component's reach for
*every* cell, so it cannot be a neighbourhood effect.

All functions are plain matplotlib on ``adata.obsm[spatial_key]`` rather than
``squidpy.pl.spatial_scatter``, because the anchor/zone colourings are categorical overlays
with a fixed order and the distance colouring is diverging around a hard zero (the border) --
both need norms and z-order that the generic scatter does not expose.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.sparse import issparse

from interscale.tl.anchors import ZONE_CORE, ZONE_FAR, ZONE_NEAR, ZONE_ORDER

__all__ = [
    "ZONE_COLORS",
    "ANCHOR_COLORS",
    "NA_COLOR",
    "anchor_map",
    "signed_distance_map",
    "zone_map",
    "gene_maps",
    "distance_profile",
]

# Core and near take the dataset palette's "closest to the SHH source" end (`#005F73` /
# `#88C8B2` in `figures/config.yml`), so a zone figure reads next to the ring figures instead
# of inventing a second distance colour language. `far` is deliberately NOT the palette's
# `#EE9B00`: it is the majority of every slide, and a saturated majority reads as the subject
# of the panel. It gets the neutral tissue tan of the fibrosis-core panels this mirrors, which
# also makes it the same colour as `outside` in :func:`anchor_map`.
ZONE_COLORS = {ZONE_CORE: "#005F73", ZONE_NEAR: "#88C8B2", ZONE_FAR: "#D9D3C4"}

# Cells with no zone at all -- a sample where the dimension found no focus, so the signed
# distance is nan for every cell of it. Zoned and unzoned cells never share a panel (a sample
# is either wholly anchored or wholly nan), so this only has to be distinguishable from the
# scale, not from the tan.
NA_COLOR = "#F0F0F0"

# Binary anchor overlay: everything that is not a focus stays the same neutral tissue tan.
ANCHOR_COLORS = {False: ZONE_COLORS[ZONE_FAR], True: ZONE_COLORS[ZONE_CORE]}


# --------------------------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------------------------


def _vector(adata, x, what):
    """Accept either an ``adata.obs`` column name or an array over cells."""
    if isinstance(x, str):
        if x not in adata.obs:
            raise KeyError(f"{x!r} not found in adata.obs")
        v = adata.obs[x]
    else:
        v = x
    v = v.to_numpy() if hasattr(v, "to_numpy") else np.asarray(v)
    if v.shape[0] != adata.n_obs:
        raise ValueError(f"{what} has length {v.shape[0]}, expected {adata.n_obs}")
    return v


def _slides(adata, sample_key, samples):
    """[(label, row indices)] for the requested samples, in the order requested."""
    grp = adata.obs[sample_key].astype(str).to_numpy()
    if samples is None:
        labels = list(map(str, pd.unique(grp)))
    else:
        labels = [str(s) for s in np.atleast_1d(samples)]
        missing = [s for s in labels if s not in set(grp)]
        if missing:
            raise KeyError(f"adata.obs['{sample_key}'] has no group(s) {missing}")
    return [(s, np.where(grp == s)[0]) for s in labels]


def _grid(n, ncols, panel_size, axes=None):
    """(fig, flat list of axes) sized so every panel is `panel_size`."""
    if axes is not None:
        axes = np.atleast_1d(np.asarray(axes, dtype=object)).ravel().tolist()
        if len(axes) < n:
            raise ValueError(f"need {n} axes, got {len(axes)}")
        return axes[0].get_figure(), axes[:n]
    ncols = n if ncols is None else min(int(ncols), n)
    nrows = int(np.ceil(n / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows), squeeze=False)
    flat = ax.ravel().tolist()
    for a in flat[n:]:
        a.set_axis_off()
    return fig, flat[:n]


def _bare(ax, title=None):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    if title is not None:
        ax.set_title(title, fontsize=10)


def _expression(adata, rows, gene, layer):
    """One gene's values for `rows`, densified for those rows only."""
    j = int(np.where(adata.var_names == gene)[0][0])
    X = adata.layers[layer] if layer is not None else adata.X
    col = X[rows, j]
    if issparse(col):
        col = col.toarray()
    return np.asarray(col, dtype=float).ravel()


# --------------------------------------------------------------------------------------------
# spatial panels
# --------------------------------------------------------------------------------------------


def anchor_map(
    adata,
    anchor,
    *,
    samples: Sequence[str] | None = None,
    sample_key: str = "sample",
    spatial_key: str = "spatial",
    focus=None,
    colors: dict | None = None,
    labels: tuple[str, str] = ("outside", "anchor core"),
    size: float = 6.0,
    ncols: int | None = None,
    panel_size: tuple[float, float] = (3.0, 3.0),
    axes=None,
    legend: bool = True,
    titles: Sequence[str] | None = None,
    show: bool = True,
):
    """Where a dimension anchors: one panel per sample, anchor foci against the rest.

    Parameters
    ----------
    anchor
        Boolean over cells, or an ``adata.obs`` column name -- what
        :func:`interscale.tl.find_anchor_cells` returned.
    focus
        Optional focus ids (its second return value). When given, each focus is numbered at
        its centroid, which is what makes "there are three anchor points" checkable rather
        than asserted.
    titles
        Panel titles, defaulting to the sample name plus its anchor-cell count.

    Returns
    -------
    (fig, axes)
    """
    anchor = _vector(adata, anchor, "anchor").astype(bool)
    focus = None if focus is None else _vector(adata, focus, "focus").astype(int)
    colors = {**ANCHOR_COLORS, **(colors or {})}
    xy = np.asarray(adata.obsm[spatial_key], dtype=float)

    panels = _slides(adata, sample_key, samples)
    fig, axs = _grid(len(panels), ncols, panel_size, axes=axes)

    for k, ((label, rows), ax) in enumerate(zip(panels, axs, strict=True)):
        inside = rows[anchor[rows]]
        outside = rows[~anchor[rows]]
        # Anchors last so a focus is never hidden under the tissue it sits in.
        ax.scatter(xy[outside, 0], xy[outside, 1], s=size, c=colors[False], linewidths=0)
        ax.scatter(xy[inside, 0], xy[inside, 1], s=size, c=colors[True], linewidths=0)
        if focus is not None:
            for f in np.unique(focus[inside]):
                cxy = xy[inside[focus[inside] == f]].mean(axis=0)
                ax.annotate(
                    str(f),
                    cxy,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
        default = f"{label}\n{inside.size} anchor cells"
        _bare(ax, default if titles is None else titles[k])

    if legend:
        axs[-1].legend(
            handles=[
                Line2D([], [], marker="o", ls="", color=colors[False], label=labels[0]),
                Line2D([], [], marker="o", ls="", color=colors[True], label=labels[1]),
            ],
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            fontsize=9,
        )
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axs


def signed_distance_map(
    adata,
    dist,
    *,
    samples: Sequence[str] | None = None,
    sample_key: str = "sample",
    spatial_key: str = "spatial",
    cmap: str = "RdBu",
    vmin: float | None = None,
    vmax: float | None = None,
    size: float = 6.0,
    ncols: int | None = None,
    panel_size: tuple[float, float] = (3.4, 3.0),
    axes=None,
    colorbar: bool = True,
    titles: Sequence[str] | None = None,
    show: bool = True,
):
    """Signed distance to the anchor border: red inside a focus, blue away from it.

    The norm is a :class:`~matplotlib.colors.TwoSlopeNorm` centred on 0 -- the border -- and
    *not* symmetric, because a focus is a few hundred um deep while the tissue outside it runs
    to several thousand: a symmetric range would flatten the core to a single shade. The
    colour scale is shared across panels so two slides are comparable.

    Returns
    -------
    (fig, axes)
    """
    dist = _vector(adata, dist, "dist").astype(float)
    xy = np.asarray(adata.obsm[spatial_key], dtype=float)

    panels = _slides(adata, sample_key, samples)
    shown = np.concatenate([rows for _, rows in panels])
    d = dist[shown]
    if not np.isfinite(d).any():
        raise ValueError("no finite signed distances in the requested samples")
    lo = float(np.nanmin(d)) if vmin is None else float(vmin)
    hi = float(np.nanmax(d)) if vmax is None else float(vmax)
    # TwoSlopeNorm insists on vmin < vcenter < vmax; a sample that is entirely outside every
    # focus has no negative side, so nudge rather than raise.
    lo = min(lo, -1e-6)
    hi = max(hi, 1e-6)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=lo, vmax=hi)

    fig, axs = _grid(len(panels), ncols, panel_size, axes=axes)
    sm = None
    for k, ((label, rows), ax) in enumerate(zip(panels, axs, strict=True)):
        # A sample with no focus is all-nan, and scatter would simply not draw it -- an empty
        # panel reads as a plotting failure rather than as the result it is, so the tissue is
        # drawn in the na colour first.
        na = rows[~np.isfinite(dist[rows])]
        if na.size:
            ax.scatter(xy[na, 0], xy[na, 1], s=size, c=NA_COLOR, linewidths=0)
        # Nearest-first so the anchor neighbourhood is not buried under distant cells.
        keep = rows[np.isfinite(dist[rows])]
        order = keep[np.argsort(-dist[keep])]
        if order.size:
            sm = ax.scatter(xy[order, 0], xy[order, 1], s=size, c=dist[order], cmap=cmap, norm=norm, linewidths=0)
        _bare(ax, label if titles is None else titles[k])

    if colorbar and sm is not None:
        cb = fig.colorbar(sm, ax=axs, fraction=0.03, pad=0.02)
        cb.set_label("signed distance to anchor border [um]", fontsize=9)
    else:
        fig.tight_layout()
    if show:
        plt.show()
    return fig, axs


def zone_map(
    adata,
    zone,
    *,
    samples: Sequence[str] | None = None,
    sample_key: str = "sample",
    spatial_key: str = "spatial",
    colors: dict | None = None,
    size: float = 6.0,
    ncols: int | None = None,
    panel_size: tuple[float, float] = (3.0, 3.0),
    axes=None,
    legend: bool = True,
    titles: Sequence[str] | None = None,
    show: bool = True,
):
    """``anchor core`` / ``near`` / ``far`` bands, one panel per sample.

    Zones are drawn in ``interscale.tl.anchors.ZONE_ORDER``, so ``core`` ends up on top
    of ``far`` and the legend order is the distance order regardless of how many cells each
    band holds. Cells with an undefined zone -- a sample where the dimension found no focus --
    are drawn in ``NA_COLOR``.

    Returns
    -------
    (fig, axes)
    """
    zone = _vector(adata, zone, "zone").astype(object)
    colors = {**ZONE_COLORS, **(colors or {})}
    xy = np.asarray(adata.obsm[spatial_key], dtype=float)

    panels = _slides(adata, sample_key, samples)
    fig, axs = _grid(len(panels), ncols, panel_size, axes=axes)

    for k, ((label, rows), ax) in enumerate(zip(panels, axs, strict=True)):
        unassigned = rows[pd.isna(zone[rows])]
        if unassigned.size:
            ax.scatter(xy[unassigned, 0], xy[unassigned, 1], s=size, c=NA_COLOR, linewidths=0)
        for z in reversed(ZONE_ORDER):  # far first, core last
            sel = rows[zone[rows] == z]
            if sel.size:
                ax.scatter(xy[sel, 0], xy[sel, 1], s=size, c=colors[z], linewidths=0)
        _bare(ax, label if titles is None else titles[k])

    if legend:
        axs[-1].legend(
            handles=[Line2D([], [], marker="o", ls="", color=colors[z], label=z) for z in ZONE_ORDER],
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            fontsize=9,
        )
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axs


def gene_maps(
    adata,
    genes: Sequence[str],
    *,
    samples: Sequence[str],
    layer: str | None = "log1p_norm",
    sample_key: str = "sample",
    spatial_key: str = "spatial",
    anchor=None,
    cmap: str = "viridis",
    vmax_quantile: float = 0.99,
    size: float = 6.0,
    panel_size: tuple[float, float] = (2.8, 2.6),
    anchor_edge: str = "#D00000",
    show: bool = True,
):
    """Expression of a dimension's top genes, genes down the rows and slides across.

    One colour scale per **gene**, taken over the slides shown (``vmax_quantile`` of the
    pooled non-zero-inclusive values), so a row compares slides. Per gene and not global
    because the genes are on very different absolute levels and a shared scale would render
    the low ones blank.

    Parameters
    ----------
    anchor
        Optional boolean/obs key. Anchor cells get a coloured ring, which is what ties the
        expression pattern back to the foci the dimension was anchored on.

    Returns
    -------
    (fig, axes) with ``axes`` shaped ``(len(genes), len(samples))``.
    """
    genes = [str(g) for g in np.atleast_1d(genes)]
    missing = [g for g in genes if g not in set(map(str, adata.var_names))]
    if missing:
        raise KeyError(f"genes not in adata.var_names: {missing}")
    anchor = None if anchor is None else _vector(adata, anchor, "anchor").astype(bool)
    xy = np.asarray(adata.obsm[spatial_key], dtype=float)

    panels = _slides(adata, sample_key, samples)
    fig, axs = plt.subplots(
        len(genes),
        len(panels),
        figsize=(panel_size[0] * len(panels) + 0.9, panel_size[1] * len(genes)),
        squeeze=False,
    )

    for i, gene in enumerate(genes):
        vals = {label: _expression(adata, rows, gene, layer) for label, rows in panels}
        pooled = np.concatenate(list(vals.values()))
        hi = float(np.quantile(pooled, vmax_quantile))
        if hi <= 0:  # gene undetected on these slides -- keep the panel honest, not blank
            hi = float(pooled.max()) or 1.0
        sm = None
        for j, (label, rows) in enumerate(panels):
            ax = axs[i, j]
            v = vals[label]
            order = np.argsort(v)  # brightest on top
            sm = ax.scatter(
                xy[rows[order], 0],
                xy[rows[order], 1],
                s=size,
                c=v[order],
                cmap=cmap,
                vmin=0.0,
                vmax=hi,
                linewidths=0,
            )
            if anchor is not None:
                a = rows[anchor[rows]]
                if a.size:
                    ax.scatter(
                        xy[a, 0],
                        xy[a, 1],
                        s=size * 2.6,
                        facecolors="none",
                        edgecolors=anchor_edge,
                        linewidths=0.35,
                    )
            _bare(ax, label if i == 0 else None)
            if j == 0:
                ax.set_ylabel(gene, fontsize=10, rotation=0, ha="right", va="center", labelpad=8)
        fig.colorbar(sm, ax=axs[i, :].tolist(), fraction=0.025, pad=0.02).set_label(layer or "X", fontsize=8)

    if show:
        plt.show()
    return fig, axs


# --------------------------------------------------------------------------------------------
# distance profiles
# --------------------------------------------------------------------------------------------


def distance_profile(
    prof: pd.DataFrame,
    *,
    genes: Sequence[str] | None = None,
    local_reach: float | None = None,
    colors: Sequence[str] | dict | None = None,
    band: bool = True,
    one_panel_per_gene: bool = False,
    ncols: int | None = 3,
    panel_size: tuple[float, float] = (3.2, 2.5),
    ax=None,
    ylabel: str = "mean expression",
    title: str | None = None,
    show: bool = True,
):
    """Binned expression against signed distance to the anchor border.

    Takes what :func:`interscale.tl.profile_by_distance` returns (``gene, center, mean, sem,
    n``). Three reference marks make the axis readable:

    * ``x = 0`` -- the anchor border. Left of it is inside a focus.
    * the shaded strip left of 0 -- the anchor core itself.
    * ``x = local_reach`` -- :func:`interscale.tl.local_reach_um`. **Anything the profile does
      to the right of this line is beyond the local component's reach**, so it cannot be a
      neighbourhood effect; that is the whole reason the profile is plotted against distance.

    Parameters
    ----------
    one_panel_per_gene
        One axes per gene (shared x). Use it when the genes differ by more than a factor of
        two in level, where a single panel hides the weaker ones' shape.

    Returns
    -------
    (fig, axes)
    """
    required = {"gene", "center", "mean", "sem"}
    if not required.issubset(prof.columns):
        raise ValueError(f"prof is missing columns {sorted(required - set(prof.columns))}")
    if prof.empty:
        raise ValueError("prof is empty -- every distance bin fell below min_cells")

    genes = list(prof["gene"].unique()) if genes is None else [str(g) for g in genes]
    missing = [g for g in genes if g not in set(prof["gene"])]
    if missing:
        raise ValueError(f"genes not present in prof: {missing}")

    if isinstance(colors, dict):
        color_of = colors
    else:
        cyc = list(colors) if colors is not None else list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        color_of = {g: cyc[i % len(cyc)] for i, g in enumerate(genes)}

    if one_panel_per_gene:
        fig, axs = _grid(len(genes), ncols, panel_size)
        panels = [(g, axs[i]) for i, g in enumerate(genes)]
    elif ax is not None:
        fig, axs = ax.get_figure(), [ax]
        panels = [(g, ax) for g in genes]
    else:
        fig, axs = plt.subplots(figsize=(panel_size[0] * 1.6, panel_size[1] * 1.3))
        axs = [axs]
        panels = [(g, axs[0]) for g in genes]

    xmin = float(prof["center"].min())
    xmax = float(prof["center"].max())

    for gene, a in panels:
        sub = prof[prof["gene"] == gene].sort_values("center")
        a.plot(sub["center"], sub["mean"], "-o", ms=3, lw=1.4, color=color_of[gene], label=gene)
        if band:
            a.fill_between(
                sub["center"],
                sub["mean"] - sub["sem"],
                sub["mean"] + sub["sem"],
                color=color_of[gene],
                alpha=0.18,
                linewidth=0,
            )

    for a in dict.fromkeys(a for _, a in panels):
        if xmin < 0:
            a.axvspan(xmin, 0.0, color=ZONE_COLORS[ZONE_CORE], alpha=0.10, linewidth=0)
        a.axvline(0.0, color="0.35", lw=1.0)
        if local_reach is not None and xmin < local_reach < xmax:
            a.axvline(local_reach, color="0.35", lw=1.0, ls="--")
            # Rotated and inside the axes: a horizontal label overruns the panel as soon as
            # the genes get one panel each, which is the layout this figure is usually in.
            a.annotate(
                f"local reach {local_reach:g} um",
                (local_reach, 0.98),
                xycoords=("data", "axes fraction"),
                xytext=(4, 0),
                textcoords="offset points",
                rotation=90,
                ha="left",
                va="top",
                fontsize=7,
                color="0.35",
            )
        a.set_xlabel("signed distance to anchor border [um]", fontsize=9)
        a.set_ylabel(ylabel, fontsize=9)
        a.spines[["top", "right"]].set_visible(False)
        if one_panel_per_gene:
            a.set_title(next(g for g, aa in panels if aa is a), fontsize=10)
        else:
            a.legend(frameon=False, fontsize=8)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axs
