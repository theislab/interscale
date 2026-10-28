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
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch

import anndata as ad

__all__ = ["COLOR_LOCAL", "COLOR_GLOBAL", "COLOR_OTHER", "top_local_global", "moran_auc", "moran_curves"]

COLOR_LOCAL = "#EE9B00"
COLOR_GLOBAL = "#337F8F"
COLOR_OTHER = "#BBBBBB"
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
    names: Sequence[str], top_local: Sequence[str], top_global: Sequence[str], color_local: str, color_global: str,
    color_other: str,
) -> list[str]:
    return [color_local if g in top_local else color_global if g in top_global else color_other for g in names]


def _finish(fig: plt.Figure, ax: Axes, save: str | Path | None, show: bool | None, own_fig: bool) -> Axes | None:
    ax.spines[["top", "right"]].set_visible(False)
    if save is not None:
        fig.savefig(save, bbox_inches="tight")
    if show is None:
        show = own_fig  # an axes passed by the caller is part of a larger figure, which the caller shows
    if show:
        plt.show()
        return None
    return ax


def moran_auc(
    adata: ad.AnnData,
    *,
    key: str = "moran_scale",
    features: Sequence[str] | None = None,
    n_top: int = 5,
    color_local: str = COLOR_LOCAL,
    color_global: str = COLOR_GLOBAL,
    color_other: str = COLOR_OTHER,
    max_labels: int = 100,
    title: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    save: str | Path | None = None,
    show: bool | None = None,
) -> Axes | None:
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
    color_local, color_global, color_other
        Bar colors of the top local, top global and remaining features.
    max_labels
        With more features than this, only the colored features are labelled on the x-axis.
    title
        Axes title.
    ax
        Axes to plot into. If ``None``, a new figure is created.
    figsize, dpi
        Size and resolution of a new figure.
    save
        Path to save the figure to.
    show
        Show the figure. Default: only if a new figure was created. If ``False``, the axes are returned.

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
    colors = _class_colors(names, top_local, top_global, color_local, color_global, color_other)

    n = len(names)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize or (min(max(3.0, 0.15 * n + 1.5), 14.0), 4.0), dpi=dpi)
    else:
        fig = ax.figure
    x = np.arange(n)
    ax.bar(x, stats["auc"], color=colors, width=0.85 if n <= max_labels else 1.0, linewidth=0)
    if n <= max_labels:
        ticks, labels = x, names
    else:
        ticks = [i for i, g in enumerate(names) if g in top_local or g in top_global]
        labels = [names[i] for i in ticks]
    many = n > 20
    ax.set_xticks(ticks, labels, rotation=90 if many else 45, ha="center" if many else "right", fontsize=6 if many else 9)
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
        frameon=False,
        fontsize=8,
        title="local index",
        title_fontsize=8,
    )
    return _finish(fig, ax, save, show, own_fig)


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
        ax.annotate(labels[i], (x, ys[i]), xytext=(x + 0.02 * (x_hi - x_lo), p), fontsize=8, color=colors[i],
                    va="center")
    ax.set_ylim(lo, max(hi, pos.max() + gap))


def moran_curves(
    adata: ad.AnnData,
    *,
    key: str = "moran_scale",
    genes: Literal["both", "local", "global"] | Sequence[str] = "both",
    features: Sequence[str] | None = None,
    n_top: int = 3,
    palette: Mapping[str, str] | None = None,
    color_local: str = COLOR_LOCAL,
    color_global: str = COLOR_GLOBAL,
    color_other: str = COLOR_OTHER,
    sd: bool = False,
    title: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (5.0, 4.0),
    dpi: int | None = None,
    save: str | Path | None = None,
    show: bool | None = None,
) -> Axes | None:
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
    color_local, color_global, color_other
        Class colors.
    sd
        Shade ± 1 standard deviation over libraries (requires ``tl.moran_scale(..., per_library=True)``).
    title
        Axes title.
    ax
        Axes to plot into. If ``None``, a new figure is created.
    figsize, dpi
        Size and resolution of a new figure.
    save
        Path to save the figure to.
    show
        Show the figure. Default: only if a new figure was created. If ``False``, the axes are returned.

    Returns
    -------
    The axes if ``show = False``, otherwise ``None``.
    """
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
    palette = {} if palette is None else palette
    class_colors = _class_colors(genes, top_local, top_global, color_local, color_global, color_other)
    colors = [palette.get(g, c) for g, c in zip(genes, class_colors)]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    for g, c in zip(genes, colors):
        ax.plot(scales, I.loc[g], marker="o", ms=4, color=c, label=g)
        if sd:
            ax.fill_between(scales, I.loc[g] - I_sd.loc[g], I.loc[g] + I_sd.loc[g], color=c, alpha=0.15, lw=0)
    ax.axvline(split, ls=":", color="grey", lw=1)
    ax.axhline(0, ls="--", lw=0.8, color="grey")
    ax.set_xlabel("Number of neighbors (k)" if scale_col == "k" else "Radius")
    ax.set_ylabel("Moran's I")
    if title is not None:
        ax.set_title(title, fontsize=10)

    in_palette = [g in palette for g in genes]
    if any(in_palette):
        handles = [h for h, keep in zip(ax.get_lines(), in_palette) if keep]
        ax.legend(handles=handles, frameon=False, fontsize=8)
    if not all(in_palette):
        # class-colored curves share a color, so they are labelled directly
        ax.set_xlim(right=scales[-1] + 0.18 * (scales[-1] - scales[0]))
        idx = [i for i, keep in enumerate(in_palette) if not keep]
        _label_line_ends(ax, scales[-1], [I.loc[genes[i]].iloc[-1] for i in idx], [genes[i] for i in idx],
                         [colors[i] for i in idx])
    return _finish(fig, ax, save, show, own_fig)
