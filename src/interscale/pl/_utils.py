"""Shared plumbing for :mod:`interscale.pl`.

Two decorators give a plotting function the parameters every one of them needs, create the axes,
and then save, show and return. The function itself only draws.

:func:`figure_frame` -- one axes, returns the axes or ``None``::

    @figure_frame(figsize=(5.0, 4.0))
    def my_plot(adata, *, color="k", frame: Frame) -> None:
        ax = frame.axes()

It adds ``ax``, ``figsize``, ``dpi``, ``save`` and ``show``.

:func:`panel_frame` -- a grid of panels, returns ``(fig, axes)``::

    @panel_frame(panel_size=(3.0, 3.0))
    def my_maps(adata, *, samples=None, frame: PanelFrame) -> None:
        for ax in frame.grid(len(samples)):
            ...

It adds ``ncols``, ``panel_size``, ``axes``, ``dpi``, ``save`` and ``show``.

Both add their parameters to the public signature, so help(), the IDE and the docs site still
show them, and write the matching documentation over the line ``{frame_params}`` in the
function's own docstring (or, with no such line, in front of its ``Returns`` section).

This is deliberately ours rather than scanpy's: scanpy repeats ``show``, ``save`` and ``ax``
in every signature of its own, its only base class (``BasePlot``) serves the grouped matrix
plots and assumes ``var_names`` x ``groupby`` categories, and its one reusable helper
(``plotting._utils.savefig_or_show``) is private, takes a filename *suffix* rather than a path,
writes into ``sc.settings.figdir``, and closes the figure after saving.
"""

from __future__ import annotations

import functools
import inspect
import textwrap
import typing
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "COLOR_LOCAL",
    "COLOR_GLOBAL",
    "COLOR_OTHER",
    "CLASS_COLORS",
    "Frame",
    "PanelFrame",
    "figure_frame",
    "panel_frame",
    "panel_grid",
    "bare_axes",
]

# Class colours for the plots that split features into local-scale and global-scale: the Moran
# plots today, anything ranking features by a local index later. `COLOR_OTHER` is the grey of
# everything that is neither, which is most of the features on a whole-panel plot.
COLOR_LOCAL = "#EE9B00"
COLOR_GLOBAL = "#337F8F"
COLOR_OTHER = "#BBBBBB"

#: The palette a plot gets by passing ``colors=CLASS_COLORS`` to its decorator; the drawing
#: function then reads ``frame.colors["local"]`` rather than declaring a parameter per colour.
CLASS_COLORS = {"local": COLOR_LOCAL, "global": COLOR_GLOBAL, "other": COLOR_OTHER}

_PLACEHOLDER = "{frame_params}"


def _colors_doc(defaults: Mapping) -> str:
    """The ``colors`` parameter's documentation, listing the palette the decorator holds."""
    shown = ", ".join(f"``{k}`` {v}" for k, v in defaults.items())
    body = textwrap.fill(f"Colour per class, merged over the defaults: {shown}.", width=104)
    return f"colors\n{textwrap.indent(body, '    ')}"


# ------------------------------------------------------------------------------------------
# figure helpers
# ------------------------------------------------------------------------------------------


def panel_grid(
    n: int,
    ncols: int | None,
    panel_size: tuple[float, float],
    *,
    axes: Sequence[Axes] | None = None,
    dpi: int | None = None,
) -> tuple[Figure, list[Axes]]:
    """``(fig, flat list of axes)`` sized so every panel is ``panel_size``.

    Parameters
    ----------
    n
        Number of panels. Spare axes of the last row are switched off.
    ncols
        Columns, or ``None`` for a single row.
    axes
        Draw into these instead of creating a figure. Must hold at least ``n`` axes.
    """
    if axes is not None:
        flat = np.atleast_1d(np.asarray(axes, dtype=object)).ravel().tolist()
        if len(flat) < n:
            raise ValueError(f"need {n} axes, got {len(flat)}")
        return flat[0].get_figure(), flat[:n]
    ncols = n if ncols is None else min(int(ncols), n)
    nrows = int(np.ceil(n / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows), squeeze=False, dpi=dpi)
    flat = ax.ravel().tolist()
    for a in flat[n:]:
        a.set_axis_off()
    return fig, flat[:n]


def bare_axes(ax: Axes, title: str | None = None) -> None:
    """A spatial panel: equal aspect, no ticks, no spines, optional title."""
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    if title is not None:
        ax.set_title(title, fontsize=10)


# ------------------------------------------------------------------------------------------
# one axes
# ------------------------------------------------------------------------------------------

FRAME_PARAMS_DOC = """\
ax
    Axes to plot into. If ``None``, a new figure is created.
figsize, dpi
    Size and resolution of a new figure.
save
    Path to save the figure to.
show
    Show the figure. Default: only if a new figure was created. If ``False``, the axes are returned.\
"""


class Frame:
    """The figure a :func:`figure_frame` function draws into, and what happens to it afterwards.

    The function receives one as ``frame``, calls :meth:`axes` to get something to draw on, and
    returns. The decorator then hides the spines, saves, shows and returns for it.
    """

    def __init__(
        self,
        *,
        ax: Axes | None,
        figsize: tuple[float, float] | None,
        dpi: int | None,
        save: str | Path | None,
        show: bool | None,
        default_figsize: tuple[float, float] | None,
        despine: Sequence[str],
        colors: dict,
    ) -> None:
        self.colors = colors
        """The palette the decorator holds, with the caller's ``colors`` merged over it."""
        self._given_ax = ax
        self._given_figsize = figsize
        self._default_figsize = default_figsize
        self._dpi = dpi
        self._save = save
        self._show = show
        self._despine = tuple(despine)
        self._ax: Axes | None = None

    @property
    def own_fig(self) -> bool:
        """Whether the figure belongs to this plot, i.e. no ``ax`` was passed in."""
        return self._given_ax is None

    def axes(self, figsize: tuple[float, float] | None = None) -> Axes:
        """The axes to draw into, created on the first call.

        Parameters
        ----------
        figsize
            Size of a new figure when the caller passed none, for a default that depends on the
            data (the width of a bar plot, say). Overrides the decorator's ``figsize``, and is
            itself overridden by the caller's.
        """
        if self._ax is None:
            if self._given_ax is not None:
                self._ax = self._given_ax
            else:
                size = self._given_figsize or figsize or self._default_figsize
                _, self._ax = plt.subplots(figsize=size, dpi=self._dpi)
        return self._ax

    def finish(self) -> Axes | None:
        """Hide the spines, save, show. Returns the axes unless the figure was shown."""
        ax = self.axes()
        if self._despine:
            ax.spines[list(self._despine)].set_visible(False)
        if self._save is not None:
            ax.figure.savefig(self._save, bbox_inches="tight")
        # An axes passed by the caller is part of a larger figure, which the caller shows.
        show = self.own_fig if self._show is None else self._show
        if show:
            plt.show()
            return None
        return ax


# ------------------------------------------------------------------------------------------
# a grid of panels
# ------------------------------------------------------------------------------------------

PANEL_PARAMS_DOC = """\
ncols
    Columns of panels. Default: one row.
panel_size
    Size of a single panel. The figure is sized from it and the number of panels.
axes
    Axes to draw the panels into. If ``None``, a new figure is created.
dpi
    Resolution of a new figure.
save
    Path to save the figure to.
show
    Show the figure. Default: only if a new figure was created.\
"""

_PANEL_PARAMS_DOC_NO_NCOLS = "\n".join(PANEL_PARAMS_DOC.splitlines()[2:])


class PanelFrame:
    """The grid a :func:`panel_frame` function draws into, and what happens to it afterwards.

    The function receives one as ``frame``, calls :meth:`grid` (or :meth:`subplots`) for the
    axes, and returns. The decorator then lays out, saves, shows, and returns ``(fig, axes)``.
    """

    def __init__(
        self,
        *,
        axes: Sequence[Axes] | None,
        ncols: int | None,
        panel_size: tuple[float, float] | None,
        dpi: int | None,
        save: str | Path | None,
        show: bool | None,
        default_ncols: int | None,
        default_panel_size: tuple[float, float],
        tight: bool,
        colors: dict,
    ) -> None:
        self.colors = colors
        """The palette the decorator holds, with the caller's ``colors`` merged over it."""
        self._given_axes = axes
        self._given_ncols = ncols
        self._given_panel_size = panel_size
        self._default_ncols = default_ncols
        self._default_panel_size = default_panel_size
        self._dpi = dpi
        self._save = save
        self._show = show
        self.tight = tight
        """Run ``fig.tight_layout()`` in :meth:`finish`. Set it to ``False`` from the function
        when it adds a colorbar, which tight_layout fights with."""
        self._fig: Figure | None = None
        self._axes: Any = None

    @property
    def own_fig(self) -> bool:
        """Whether the figure belongs to this plot, i.e. no ``axes`` were passed in."""
        return self._given_axes is None

    @property
    def panel_size(self) -> tuple[float, float]:
        """The panel size in force: the caller's, else the decorator's."""
        return self._given_panel_size or self._default_panel_size

    @property
    def fig(self) -> Figure:
        """The figure, once :meth:`grid` or :meth:`subplots` has been called."""
        if self._fig is None:
            raise RuntimeError("call `frame.grid(...)` or `frame.subplots(...)` before `frame.fig`.")
        return self._fig

    def grid(self, n: int, *, ncols: int | None = None, panel_size: tuple[float, float] | None = None) -> list[Axes]:
        """``n`` panels row-major, as a flat list. The caller's ``ncols``/``panel_size`` win."""
        cols = self._given_ncols if self._given_ncols is not None else (ncols or self._default_ncols)
        self._fig, self._axes = panel_grid(
            n,
            cols,
            self._given_panel_size or panel_size or self._default_panel_size,
            axes=self._given_axes,
            dpi=self._dpi,
        )
        return self._axes

    def subplots(self, nrows: int, ncols: int, *, figsize: tuple[float, float]) -> np.ndarray:
        """A fixed ``nrows`` x ``ncols`` grid, kept 2-D, for a layout the panel size does not set."""
        if self._given_axes is not None:
            arr = np.asarray(self._given_axes, dtype=object).reshape(nrows, ncols)
            self._fig, self._axes = arr.flat[0].get_figure(), arr
        else:
            self._fig, self._axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=self._dpi, squeeze=False)
        return self._axes

    def finish(self) -> tuple[Figure, Any]:
        """Lay out, save, show. Returns ``(fig, axes)`` either way."""
        fig = self.fig
        if self.tight:
            fig.tight_layout()
        if self._save is not None:
            fig.savefig(self._save, bbox_inches="tight")
        # Axes passed by the caller are part of a larger figure, which the caller shows.
        show = self.own_fig if self._show is None else self._show
        if show:
            plt.show()
        return fig, self._axes


# ------------------------------------------------------------------------------------------
# the decorators
# ------------------------------------------------------------------------------------------

_FRAME_ANNOTATIONS: dict[str, object] = {
    "ax": Axes | None,
    "figsize": tuple[float, float] | None,
    "dpi": int | None,
    "save": str | Path | None,
    "show": bool | None,
}

_PANEL_ANNOTATIONS: dict[str, object] = {
    "ncols": int | None,
    "panel_size": tuple[float, float] | None,
    "axes": "Sequence[Axes] | None",
    "dpi": int | None,
    "save": str | Path | None,
    "show": bool | None,
}

_UNSET = object()


def _public_signature(func: Callable, added: dict[str, object], return_annotation: object) -> inspect.Signature:
    """`func`'s signature without ``frame`` and with `added`, so help() shows the real API."""
    sig = inspect.signature(func)
    params = [p for name, p in sig.parameters.items() if name != "frame"]
    clash = sorted({p.name for p in params} & set(added))
    if clash:
        raise TypeError(f"{func.__qualname__}: the decorator already provides {clash}.")
    kind = inspect.Parameter.KEYWORD_ONLY
    new = [inspect.Parameter(n, kind, default=None, annotation=a) for n, a in added.items()]
    var_kw = [p for p in params if p.kind is inspect.Parameter.VAR_KEYWORD]
    head = [p for p in params if p.kind is not inspect.Parameter.VAR_KEYWORD]
    sig = sig.replace(parameters=head + new + var_kw)
    return sig if return_annotation is _UNSET else sig.replace(return_annotation=return_annotation)


def _public_doc(doc: str | None, block: str) -> str | None:
    """Substitute ``{frame_params}``, or insert `block` in front of the ``Returns`` section."""
    if not doc:
        return doc
    lines = doc.splitlines()
    for line in lines:
        if _PLACEHOLDER in line:
            indent = line[: len(line) - len(line.lstrip())]
            return doc.replace(line, textwrap.indent(block, indent))
    for i in range(1, len(lines)):
        if lines[i - 1].strip() == "Returns" and set(lines[i].strip()) == {"-"}:
            head = lines[i - 1]
            indent = head[: len(head) - len(head.lstrip())]
            j = i - 1
            while j > 0 and not lines[j - 1].strip():  # keep the blank line before `Returns`
                j -= 1
            return "\n".join(lines[:j] + textwrap.indent(block, indent).splitlines() + lines[j:])
    return f"{doc}\n{textwrap.indent(block, '    ')}\n"


def _public_annotations(func: Callable, added: dict[str, object], return_annotation: object) -> dict[str, object]:
    try:  # resolved here, where `func`'s module globals are the ones that can resolve them
        hints = typing.get_type_hints(func)
    except (NameError, TypeError):  # an annotation that does not resolve is not worth an ImportError
        hints = dict(getattr(func, "__annotations__", {}))
    hints.pop("frame", None)
    hints = {**hints, **added}
    if return_annotation is not _UNSET:
        hints["return"] = return_annotation
    return hints


def _check_frame_param(func: Callable) -> None:
    if "frame" not in inspect.signature(func).parameters:
        raise TypeError(f"{func.__qualname__} must take a keyword-only `frame` parameter.")


def figure_frame(
    *,
    figsize: tuple[float, float] | None = None,
    despine: Sequence[str] = ("top", "right"),
    colors: Mapping | None = None,
) -> Callable[[Callable], Callable]:
    """Give a one-axes plot its ``ax``, ``figsize``, ``dpi``, ``save`` and ``show`` parameters.

    The decorated function takes a keyword-only ``frame`` parameter instead of those five, calls
    ``frame.axes()`` for something to draw on, and returns nothing. The wrapper then hides the
    spines, saves, shows, and returns the axes or ``None``.

    Parameters
    ----------
    figsize
        Default size of a new figure. A function whose default depends on the data passes its
        own to :meth:`Frame.axes` instead; the caller's ``figsize`` beats both.
    despine
        Spines to hide once the function has drawn. Pass ``()`` to keep all four.
    colors
        The plot's palette, e.g. :data:`CLASS_COLORS`. The function reads it as
        ``frame.colors`` instead of taking a parameter per colour, and the caller gets one
        ``colors`` parameter that is merged over it key by key.

    Returns
    -------
    The decorator.
    """
    default_figsize = figsize  # the wrapper's own `figsize` parameter shadows the name below
    palette = dict(colors or {})
    added = {**({"colors": "Mapping | None"} if colors else {}), **_FRAME_ANNOTATIONS}
    block = FRAME_PARAMS_DOC if not colors else f"{_colors_doc(palette)}\n{FRAME_PARAMS_DOC}"

    def decorate(func: Callable) -> Callable:
        _check_frame_param(func)

        @functools.wraps(func)
        def wrapper(*args, ax=None, figsize=None, dpi=None, save=None, show=None, colors=None, **kwargs):
            frame = Frame(
                colors={**palette, **(colors or {})},
                ax=ax,
                figsize=figsize,
                dpi=dpi,
                save=save,
                show=show,
                default_figsize=default_figsize,
                despine=despine,
            )
            func(*args, frame=frame, **kwargs)
            return frame.finish()

        wrapper.__signature__ = _public_signature(func, added, Axes | None)
        wrapper.__doc__ = _public_doc(func.__doc__, block)
        wrapper.__annotations__ = _public_annotations(func, added, Axes | None)
        return wrapper

    return decorate


def panel_frame(
    *,
    panel_size: tuple[float, float],
    ncols: int | None = None,
    columns: bool = True,
    tight: bool = True,
    colors: Mapping | None = None,
) -> Callable[[Callable], Callable]:
    """Give a multi-panel plot its ``ncols``, ``panel_size``, ``axes``, ``dpi``, ``save`` and ``show``.

    The decorated function takes a keyword-only ``frame`` parameter instead of those, calls
    ``frame.grid(n)`` or ``frame.subplots(...)`` for the axes, and returns nothing. The wrapper
    then lays out, saves, shows, and returns ``(fig, axes)``.

    Parameters
    ----------
    panel_size
        Default size of a single panel.
    ncols
        Default number of columns. ``None`` puts every panel in one row.
    columns
        Whether the plot has a free column count. ``False`` leaves ``ncols`` out of the
        signature, for a layout whose grid is fixed by the data (genes x samples, say).
    tight
        Whether to run ``fig.tight_layout()`` afterwards. The function can override it on the
        frame, which is what a plot that adds a colorbar only sometimes needs.
    colors
        The plot's palette, e.g. ``ZONE_COLORS``. The function reads it as ``frame.colors``
        instead of taking a parameter per colour, and the caller gets one ``colors`` parameter
        that is merged over it key by key.

    Returns
    -------
    The decorator.
    """
    default_panel_size, default_ncols, default_tight = panel_size, ncols, tight
    palette = dict(colors or {})
    added = dict(_PANEL_ANNOTATIONS) if columns else {k: v for k, v in _PANEL_ANNOTATIONS.items() if k != "ncols"}
    block = PANEL_PARAMS_DOC if columns else _PANEL_PARAMS_DOC_NO_NCOLS
    if colors:
        added = {"colors": "Mapping | None", **added}
        block = f"{_colors_doc(palette)}\n{block}"

    def decorate(func: Callable) -> Callable:
        _check_frame_param(func)

        @functools.wraps(func)
        def wrapper(
            *args, panel_size=None, axes=None, dpi=None, save=None, show=None, ncols=None, colors=None, **kwargs
        ):
            if ncols is not None and not columns:
                raise TypeError(f"{func.__name__}() has a fixed column count; `ncols` does not apply.")
            frame = PanelFrame(
                colors={**palette, **(colors or {})},
                axes=axes,
                ncols=ncols,
                panel_size=panel_size,
                dpi=dpi,
                save=save,
                show=show,
                default_ncols=default_ncols,
                default_panel_size=default_panel_size,
                tight=default_tight,
            )
            func(*args, frame=frame, **kwargs)
            return frame.finish()

        wrapper.__signature__ = _public_signature(func, added, _UNSET)
        wrapper.__doc__ = _public_doc(func.__doc__, block)
        wrapper.__annotations__ = _public_annotations(func, added, _UNSET)
        return wrapper

    return decorate
