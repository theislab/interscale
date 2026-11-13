"""Plots of the scale-dependent Moran analysis (``tl.moran_scale``).

Both plots read ``adata.uns[key]`` and highlight the same genes: the ``n_top`` features with the highest local index
(local) and the ``n_top`` features with the lowest local index (global), ranked among ``features``.

    moran_auc(adata, n_top=5)                        # bars: AUC per gene, top local / global colored
    moran_curves(adata, genes="both", n_top=3)       # Moran's I vs. k of the top local and global genes
    moran_curves(adata, genes=["SHH", "NKX6.1"])     # ... or of a gene list
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import anndata as ad
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from interscale.pl._utils import CLASS_COLORS, COLOR_GLOBAL, COLOR_LOCAL, COLOR_OTHER, Frame, figure_frame

__all__ = ["COLOR_LOCAL", "COLOR_GLOBAL", "COLOR_OTHER", "top_local_global", "moran_auc", "moran_curves"]

MAX_CURVES = 20  # more curves than this are hard to read


def _results(adata: ad.AnnData, key: str) -> dict:
    if key not in adata.uns:
        raise KeyError(f"`adata.uns[{key!r}]` not found, run `tl.moran_scale(adata, key_added={key!r})` first.")
    return adata.uns[key]


def top_local_global(
    adata: ad.AnnData, *, key: str = "moran_scale", n_top: int = 5, features: Sequence[str] | None = None
) -> tuple[list[str], list[str]]:
    """Features with the highest (local) and lowest (global) local index.

    Parameters
    ----------
    adata
        Annotated data with ``adata.uns[key]`` from ``tl.moran_scale``.
    key
        Key of the results in ``adata.uns``.
    n_top
        Number of local and of global features. At most half of the ranked features each, so the two never overlap.
    features
        Rank only these features (default: all).

    Returns
    -------
    Top local features (most local first) and top global features (most global first).
    """
    stats = _results(adata, key)["stats"].set_index("feature")
    if features is not None:
        stats = stats.loc[list(features)]
    ranked = stats["local_index"].dropna().sort_values(ascending=False).index.tolist()
    n_top = min(n_top, len(ranked) // 2)
    return ranked[:n_top], ranked[::-1][:n_top]


def _class_colors(
    names: Sequence[str], top_local: Sequence[str], top_global: Sequence[str], colors: Mapping[str, str]
) -> list[str]:
    local, glob, other = colors["local"], colors["global"], colors["other"]
    return [local if g in top_local else glob if g in top_global else other for g in names]


@figure_frame(colors=CLASS_COLORS)
def moran_auc(
    adata: ad.AnnData,
    *,
    key: str = "moran_scale",
    features: Sequence[str] | None = None,
    n_top: int = 5,
    max_labels: int = 100,
    title: str | None = None,
    legend_kw: Mapping[str, object] | None = None,
    frame: Frame,
) -> None:
    """Bar plot of the AUC of the Moran curve per feature, sorted by AUC, top local / global features colored.

    The AUC only orders the bars; the colors come from the local index (see :func:`top_local_global`).

    Parameters
    ----------
    adata
        Annotated data with ``adata.uns[key]`` from ``tl.moran_scale``.
    key
        Key of the results in ``adata.uns``.
    features
        Features to show and rank (default: all, e.g. all genes).
    n_top
        Number of top local and of top global features to color.
    max_labels
        With more features than this, only the colored features are labelled on the x-axis.
    title
        Axes title.
    legend_kw
        Extra arguments for ``ax.legend``, e.g. ``{"loc": "center left", "bbox_to_anchor": (1.0, 0.5)}``
        to put the legend outside the axes, to the right.
    {frame_params}

    Returns
    -------
    The axes if ``show = False``, otherwise ``None``.
    """
    stats = _results(adata, key)["stats"].set_index("feature")
    if features is not None:
        stats = stats.loc[list(features)]
    top_local, top_global = top_local_global(adata, key=key, n_top=n_top, features=stats.index)
    stats = stats.sort_values("auc", ascending=False)
    names = stats.index.tolist()
    color_local, color_global = frame.colors["local"], frame.colors["global"]
    colors = _class_colors(names, top_local, top_global, frame.colors)

    n = len(names)
    # The width grows with the number of bars, so a 40-gene panel is not 400 genes squeezed.
    ax = frame.axes(figsize=(min(max(3.0, 0.15 * n + 1.5), 14.0), 4.0))
    x = np.arange(n)
    ax.bar(x, stats["auc"], color=colors, width=0.85 if n <= max_labels else 1.0, linewidth=0)
    if n <= max_labels:
        ticks, labels = x, names
    else:
        ticks = [i for i, g in enumerate(names) if g in top_local or g in top_global]
        labels = [names[i] for i in ticks]
    many = n > 20
    ax.set_xticks(
        ticks, labels, rotation=90 if many else 45, ha="center" if many else "right", fontsize=6 if many else 9
    )
    for tick, g in zip(ax.get_xticklabels(), labels):
        if g in top_local or g in top_global:
            tick.set_color(color_local if g in top_local else color_global)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylabel("AUC of Moran's I curve")
    if title is not None:
        ax.set_title(title, fontsize=10)
    ax.legend(
        handles=[
            Patch(color=color_local, label=f"top {len(top_local)} local"),
            Patch(color=color_global, label=f"top {len(top_global)} global"),
        ],
        **{"frameon": False, "fontsize": 8, "title": "local index", "title_fontsize": 8, **(legend_kw or {})},
    )


def _label_line_ends(ax: Axes, x: float, ys: Sequence[float], labels: Sequence[str], colors: Sequence[str]) -> None:
    """Label curves at their right end; labels are pushed apart vertically so they do not overlap."""
    order = np.argsort(ys)
    lo, hi = ax.get_ylim()
    x_lo, x_hi = ax.get_xlim()
    gap = 0.05 * (hi - lo)
    pos = np.asarray(ys, dtype=float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    for p, i in zip(pos, order):
        ax.annotate(
            labels[i], (x, ys[i]), xytext=(x + 0.02 * (x_hi - x_lo), p), fontsize=8, color=colors[i], va="center"
        )
    ax.set_ylim(lo, max(hi, pos.max() + gap))


@figure_frame(figsize=(5.0, 4.0), colors=CLASS_COLORS)
def moran_curves(
    adata: ad.AnnData,
    *,
    key: str = "moran_scale",
    genes: Literal["both", "local", "global"] | Sequence[str] = "both",
    features: Sequence[str] | None = None,
    n_top: int = 3,
    palette: Mapping[str, str] | None = None,
    sd: bool = False,
    log_scale: bool = False,
    chord: bool = False,
    normalize: bool = False,
    title: str | None = None,
    legend_kw: Mapping[str, object] | None = None,
    frame: Frame,
) -> None:
    """Moran curves: Moran's I against the number of neighbors ``k`` (or the radius).

    Parameters
    ----------
    adata
        Annotated data with ``adata.uns[key]`` from ``tl.moran_scale``.
    key
        Key of the results in ``adata.uns``.
    genes
        ``"both"`` (default): the top ``n_top`` local and top ``n_top`` global features; ``"local"`` / ``"global"``:
        only those; or a list of features. Lists of more than ``MAX_CURVES`` features are hard to read.
    features
        Rank the local / global features among these only (default: all), as in :func:`moran_auc`.
    n_top
        Number of top local and of top global features.
    palette
        Color per feature, with a legend. Features not in ``palette`` (or all, if ``None``) are colored by class:
        top local, top global or other, and labelled at the end of their curve.
    sd
        Shade ± 1 standard deviation over libraries (requires ``tl.moran_scale(..., per_library=True)``).
    log_scale
        Plot against ``log2`` of the scale. On this axis a decay with no preferred scale (the same ΔI per doubling
        of ``k``) is a straight line.
    chord
        Draw each curve's chord, the straight line from ``I(k_min)`` to ``I(k_max)`` on the log2 axis, and the gap
        between the chord and ``I(k_split)``: a curve sagging below its chord (convex) loses its autocorrelation
        within small neighborhoods (local pattern), one bulging above it (concave) at tissue scale. The gap is the
        stats column ``chord_gap``, a rescaled local index (see ``tl.delta_from_profile``). ``k_min``, ``k_split``
        and ``k_max`` are drawn filled, the scales the index does not use hollow. Requires ``log_scale``.
    normalize
        Rescale each curve to ``(I - I(k_max)) / (I(k_min) - I(k_max))``, so every curve runs from 1 to 0 and all
        share one chord (the diagonal). Makes the curvature of strongly and weakly autocorrelated features
        comparable.
    title
        Axes title.
    legend_kw
        Extra arguments for ``ax.legend``, e.g. ``{"loc": "center left", "bbox_to_anchor": (1.0, 0.5)}``
        to put the legend outside the axes, to the right. Only curves colored by ``palette`` get a legend
        entry; class-colored curves are labelled at their right end instead.
    {frame_params}

    Returns
    -------
    The axes if ``show = False``, otherwise ``None``.
    """
    if chord and not log_scale:
        raise ValueError("`chord=True` requires `log_scale=True`: the chord is the no-preferred-scale line on log2 k.")
    res = _results(adata, key)
    params = res["params"]
    scale_col, split = params["scale_col"], params["split"]
    top_local, top_global = top_local_global(adata, key=key, n_top=n_top, features=features)
    if isinstance(genes, str):
        if genes not in ("both", "local", "global"):
            raise ValueError(f"`genes` must be 'both', 'local', 'global' or a list of features, got {genes!r}.")
        genes = {"both": top_local + top_global, "local": top_local, "global": top_global}[genes]
    genes = list(genes)
    missing = set(genes) - set(res["stats"]["feature"])
    if missing:
        raise KeyError(f"Features not in `adata.uns[{key!r}]`: {sorted(missing)}.")
    if len(genes) > MAX_CURVES:
        warnings.warn(f"Plotting {len(genes)} curves; consider fewer than {MAX_CURVES}.", stacklevel=2)

    I = res["profile"].pivot(index="feature", columns=scale_col, values="I")
    scales = I.columns.to_numpy(dtype=float)
    if sd:
        if "profile_per_library" not in res:
            raise ValueError("`sd=True` requires `tl.moran_scale(..., per_library=True)`.")
        I_sd = res["profile_per_library"].pivot_table(index="feature", columns=scale_col, values="I", aggfunc="std")
        I_sd = I_sd.reindex(index=I.index, columns=I.columns)
    if normalize:
        span = I.iloc[:, 0] - I.iloc[:, -1]
        flat = [g for g in genes if not abs(span[g]) > 1e-12]
        if flat:
            raise ValueError(f"Cannot normalize features with I(k_min) == I(k_max): {flat}.")
        I = I.sub(I.iloc[:, -1], axis=0).div(span, axis=0)
        if sd:
            I_sd = I_sd.div(span.abs(), axis=0)
    x = np.log2(scales) if log_scale else scales
    split_x = np.log2(split) if log_scale else split
    i_split = int(np.argmin(np.abs(scales - split)))
    palette = {} if palette is None else palette
    class_colors = _class_colors(genes, top_local, top_global, frame.colors)
    colors = [palette.get(g, c) for g, c in zip(genes, class_colors)]

    ax = frame.axes()
    used = np.zeros(len(scales), dtype=bool)
    used[[0, i_split, -1]] = True  # the three scales the local index is computed from
    curves = []
    for g, c in zip(genes, colors):
        y = I.loc[g].to_numpy(dtype=float)
        if chord:
            curves.append(ax.plot(x, y, color=c, label=g)[0])
            ax.plot(x[used], y[used], "o", ms=5, color=c)
            ax.plot(x[~used], y[~used], "o", ms=4, mfc="white", mec=c)
            y_chord = y[0] + (y[-1] - y[0]) * (split_x - x[0]) / (x[-1] - x[0])
            if not normalize:
                ax.plot(x[[0, -1]], y[[0, -1]], ls="--", lw=0.9, color=c, alpha=0.7)
            ax.plot([split_x, split_x], [y_chord, y[i_split]], lw=3, color=c, alpha=0.5, solid_capstyle="butt")
        else:
            curves.append(ax.plot(x, y, marker="o", ms=4, color=c, label=g)[0])
        if sd:
            s_ = I_sd.loc[g].to_numpy(dtype=float)
            ax.fill_between(x, y - s_, y + s_, color=c, alpha=0.15, lw=0)
    if chord and normalize:
        ax.plot(x[[0, -1]], [1, 0], ls="--", lw=1, color="grey")
    ax.axvline(split_x, ls=":", color="grey", lw=1)
    ax.axhline(0, ls="--", lw=0.8, color="grey")
    name = "Number of neighbors (k)" if scale_col == "k" else "Radius"
    if log_scale:
        ax.set_xticks(x, [f"{v:g}" for v in scales])
        name += ", log2 scale"
    ax.set_xlabel(name)
    ax.set_ylabel("Moran's I, rescaled (1 at k_min, 0 at k_max)" if normalize else "Moran's I")
    if title is not None:
        ax.set_title(title, fontsize=10)

    in_palette = [g in palette for g in genes]
    handles = [h for h, keep in zip(curves, in_palette) if keep]
    if chord:
        handles += [
            Line2D([], [], ls="--", color="grey", label="chord: no preferred scale"),
            Line2D([], [], lw=3, color="grey", alpha=0.5, label="gap = chord − I(k_split)"),
            Line2D([], [], ls="", marker="o", mfc="white", mec="grey", label="not used in the index"),
        ]
    if handles:
        ax.legend(handles=handles, **{"frameon": False, "fontsize": 8, **(legend_kw or {})})
    if not all(in_palette):
        # class-colored curves share a color, so they are labelled directly
        ax.set_xlim(right=x[-1] + 0.18 * (x[-1] - x[0]))
        idx = [i for i, keep in enumerate(in_palette) if not keep]
        _label_line_ends(
            ax, x[-1], [I.loc[genes[i]].iloc[-1] for i in idx], [genes[i] for i in idx], [colors[i] for i in idx]
        )
