"""
Over-squashing: balanced Forman curvature of every edge, for each graph type

Curvature of the edge between cells i and j (Topping et al. 2022, Definition 1):

Parts with oytputs in results/oversquashing/

1. Balanced Forman curvature for every edge of every graph 1. Balanced Forman curvature of every edge of every graph:
    edges_<graph>.csv, cells_curvature.csv
2. Distribution per graph, and curvature vs edge length:
    curvature_distribution.png, curvature_summary.csv
3. Curvature on the tissue of one sample:
    curvature_tissue_<sample>.png
4. Error on hidden cells vs the cell's curvature, per GNN
    cells_error.csv, error_vs_curvature.csv/.png
5. One table comparing the graphs and the GNNs
"""

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import networkx as nx
import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import NullFormatter, ScalarFormatter
from scipy.stats import rankdata, spearmanr

from interscale.config import load_config
from interscale.model import LocalModel
from interscale.tl.geome_utils import prepare_a2d_dataset
from interscale.tl.masking import MASK_VALUE
from interscale.tl.utils import get_model_filename_prefix

DATA = Path("data/legnini_prepared.h5ad")
GRAPHS = ["radius", "knn", "delaunay"]
GNNS = ["GCN", "GIN", "GAT"]
CONFIG = "configs/legnini_{gnn}_{graph}.yaml"
MAP_SAMPLE = "slide4_A2-3"  # part of the test sample
# this is for the part 3 to draw the curvature map for one tissue section
EVAL_SPLITS = ["val", "test"]
N_ROUNDS = 3  # one round is shuffle the cells and cutting them into 5 groups of 20% each, hides one group at a time
# by shuffle = which cells get hidden in which step
SEED = 0
OUT = Path("results/oversquashing")


GRAPH_COLOUR = {"radius": "#2a78d6", "knn": "#eb6834", "delaunay": "#1baf7a"}
MODEL_STYLE = {  # colour, marker, line style
    "GCN": ("#6250d6", "o", "-"),
    "GIN": ("#e87ba4", "s", "-"),
    "GAT": ("#008300", "^", "-"),
    "neighbour mean": ("#52514e", "D", "--"),  # baseline to predict a hidden cell (regression task = gene )
}

CURVATURE_CMAP = LinearSegmentedColormap.from_list(
    "curvature",
    [
        (0.0, "#0d366b"),
        (0.25, "#2a78d6"),
        (0.4, "#86b6ef"),
        (0.5, "#d4d3cf"),
        (0.6, "#f2a49e"),
        (0.75, "#e34948"),
        (1.0, "#8f1f1f"),
    ],
)

# coming from the paper
# Definition 1, equation 3
CURVATURE_NORM = Normalize(-2, 2)  # not sure about it = slide
# every edge gets a curvature value = some number between roughlyt -2 and +1.5
# The paper states the floor explicitly. Right after Definition 1 it says that Ric(i,j) > −2


def fmt(v):
    """Numbers for the printed tables: counts with thousands separators, the rest with 3 significant digits."""
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#52514e",
        "axes.labelcolor": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

# 1. Compute the balanced Forman curvature for every edge for each graph type (kNN, radius, Delaunay)
#   Ric(i, j) = 2/d_i + 2/d_j - 2                                tree-like part: more neighbours push it towards -2
#             + 2 |T| / max(d_i, d_j) + |T| / min(d_i, d_j)      triangles raise it
#             + (|Q_i| + |Q_j|) / (gamma_max * max(d_i, d_j))    squares raise it
#   T          cells that are neighbours of both i and j (each one closes a triangle with the edge)
#   Q_i        neighbours k of i (k is not j and not a neighbour of j) for which some cell w closes the square
#              i - k - w - j without diagonals: w is a neighbour of k and of j, w is not i and not a neighbour of i
#   Q_j        the same, seen from j
#   gamma_max  the largest number of such squares that pass through one cell k (or w); no squares -> no square term
#   Ric(i, j) = 0 if i or j has only one neighbour


def undirected(adj):
    """
    Coming from Defintion 1 in the paper.
    The graph the curvature is defined on: undirected, unweighted, without self-loops.
    Every stored entry counts as an edge, as when geome turns the matrix into the GNN's edge_index.
    Making sure that each stores graph is undirected, unweighted, without self-loops as it is the one defined in the paper
    """
    # making a new matrix in csr layout
    a = sp.csr_matrix(adj, copy=True)
    # setting every nonzero entry to 1 = to make it unweghted (potential issue?)
    a.data = np.ones_like(a.data, dtype=np.float32)
    # paper assume an undirected graph, so making sure it
    a = (a + a.T).tocsr()
    # setting value to every diagonal position to zero = definition 1 requires a simple graph, so no self-loops
    a.setdiag(0)
    # deleting every entry whose value is 0 = just removing self-loops
    a.eliminate_zeros()
    # writes 1 into every stored value
    a.data[:] = 1
    # sorts the column numbers inside each row
    a.sort_indices()
    return a


@numba.njit(cache=True)  # parallelizing
def _curvature(indptr, indices):
    """Ric and gamma_max of every stored entry (i, j) of a symmetric CSR matrix, in storage order.
    Computes the curvature of every edge with the formula from Definition 1 2022 paper.
    """
    n = indptr.size - 1
    ric = np.zeros(indices.size)
    gamma_max = np.zeros(indices.size, np.int64)
    nb_i = np.zeros(n, np.int64)  # nb_i[v] == i + 1: v is a neighbour of i
    nb_j = np.zeros(n, np.int64)  # nb_j[v] == p + 1: v is a neighbour of j, for the edge stored at position p
    for i in range(n):
        d_i = indptr[i + 1] - indptr[i]
        for p in range(indptr[i], indptr[i + 1]):
            nb_i[indices[p]] = i + 1
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            d_j = indptr[j + 1] - indptr[j]
            if min(d_i, d_j) == 1:
                continue  # Ric = 0 by definition
            for q in range(indptr[j], indptr[j + 1]):
                nb_j[indices[q]] = p + 1
            triangles = 0
            for q in range(indptr[i], indptr[i + 1]):
                if nb_j[indices[q]] == p + 1:
                    triangles += 1
            # squares seen from i: k next to i (not j, not next to j), w next to k and to j (not i, not next to i)
            squares_i = 0
            gamma = 0
            for q in range(indptr[i], indptr[i + 1]):
                k = indices[q]
                if k == j or nb_j[k] == p + 1:
                    continue
                n_w = 0
                for r in range(indptr[k], indptr[k + 1]):
                    w = indices[r]
                    if w != i and nb_j[w] == p + 1 and nb_i[w] != i + 1:
                        n_w += 1
                if n_w > 0:
                    squares_i += 1
                    gamma = max(gamma, n_w)
            # the same seen from j
            squares_j = 0
            for q in range(indptr[j], indptr[j + 1]):
                w = indices[q]
                if w == i or nb_i[w] == i + 1:
                    continue
                n_k = 0
                for r in range(indptr[w], indptr[w + 1]):
                    k = indices[r]
                    if k != j and nb_i[k] == i + 1 and nb_j[k] != p + 1:
                        n_k += 1
                if n_k > 0:
                    squares_j += 1
                    gamma = max(gamma, n_k)
            d_max = max(d_i, d_j)
            d_min = min(d_i, d_j)
            value = 2 / d_i + 2 / d_j - 2 + 2 * triangles / d_max + triangles / d_min
            if gamma > 0:
                value += (squares_i + squares_j) / (gamma * d_max)
            ric[p] = value
            gamma_max[p] = gamma
    return ric, gamma_max


def edge_curvature(adj):
    """Curvature of every edge, once per pair of cells (i < j), and whether it meets Theorem 4.
    This turns the output of curvature into a table with one row per edge, and it marks the edges that meet the bottleneck
    condition (Theorem 4).
    """
    a = undirected(adj)
    ric, gamma = _curvature(a.indptr, a.indices)
    degree = np.diff(a.indptr)
    i = np.repeat(np.arange(a.shape[0]), degree)
    j = a.indices
    once = i < j  # the symmetric matrix holds every pair twice, with the same values
    i, j, ric, gamma = i[once], j[once], ric[once], gamma[once]
    d_max = np.maximum(degree[i], degree[j])
    # Theorem 4 applies if some delta exists with Ric + 2 <= delta < 1/sqrt(d_max) and delta < 1/gamma_max
    limit = np.minimum(1 / np.sqrt(d_max), np.where(gamma > 0, 1 / np.maximum(gamma, 1), np.inf))
    edges = pd.DataFrame({"i": i, "j": j, "curvature": ric, "gamma_max": gamma, "d_i": degree[i], "d_j": degree[j]})
    edges["bottleneck"] = ric + 2 < limit
    return edges, degree


def check_against_paper():
    """Reproduce the curvature of simple graphs (Topping et al. 2022, Table 1), so a wrong formula cannot slip by.
    Test that runs at the start of the script. Builds small graphs whose curvature is known exactly.
    Stops with an error if the value is off.
    """

    def inner_edges(graph, degree):
        # edges whose cells, and all cells within two hops, have the full degree: away from the border
        deep = {
            v
            for v in graph
            if all(graph.degree(u) == degree for u in nx.single_source_shortest_path_length(graph, v, cutoff=2))
        }
        return {(min(a, b), max(a, b)) for a, b in graph.edges if a in deep and b in deep}

    tree = nx.balanced_tree(2, 7)
    grid = nx.convert_node_labels_to_integers(nx.grid_2d_graph(10, 10))
    lattice = nx.convert_node_labels_to_integers(nx.triangular_lattice_graph(16, 16))
    cases = [
        ("clique K5", nx.complete_graph(5), None, 5 / 4),
        ("cycle C4", nx.cycle_graph(4), None, 1.0),
        ("cycle C6", nx.cycle_graph(6), None, 0.0),
        ("binary tree", tree, inner_edges(tree, 3), 4 / 3 - 2),
        ("square grid", grid, inner_edges(grid, 4), 0.0),
        ("even triangulation", lattice, inner_edges(lattice, 6), -1 / 3),
    ]
    for name, graph, keep, expected in cases:
        edges, _ = edge_curvature(nx.to_scipy_sparse_array(graph, nodelist=range(len(graph)), format="csr"))
        if keep is not None:
            edges = edges[[pair in keep for pair in zip(edges.i, edges.j)]]
        assert len(edges) > 0 and np.allclose(edges.curvature, expected), f"curvature is wrong on the {name}"
    print("curvature check: matches Topping et al. 2022 (Table 1) on cliques, cycles, trees, grids, triangulations")


def cell_curvature(edges, n_cells):
    """Per cell: mean and lowest curvature of its edges, and whether one of them is a bottleneck.
    It runs per-edge curvature into per-cell values. Needed for part 4
    """
    ends = np.concatenate([edges.i, edges.j])
    ric = np.concatenate([edges.curvature, edges.curvature])
    n_edges = np.bincount(ends, minlength=n_cells)
    mean = np.bincount(ends, weights=ric, minlength=n_cells) / np.where(n_edges > 0, n_edges, np.nan)
    lowest = np.full(n_cells, np.nan)
    np.fmin.at(lowest, ends, ric)
    at_bottleneck = np.zeros(n_cells, bool)
    at_bottleneck[np.concatenate([edges.i[edges.bottleneck], edges.j[edges.bottleneck]])] = True
    return mean, lowest, at_bottleneck


def part1(adata):
    """Curvature of every edge (saved per graph) and per cell (one table for all graphs).
    Runs the curvature for each of the stored graphs and saves two kinds of tables:
    one row per edge, and one row per cell
    """
    xy = adata.obsm["spatial"]
    edges_per_graph, cells = {}, []
    for graph in GRAPHS:
        adj = adata.obsp[f"{graph}_connectivities"]
        edges, degree = edge_curvature(adj)
        edges["length"] = np.linalg.norm(xy[edges.i] - xy[edges.j], axis=1)
        edges["sample"] = adata.obs["sample"].to_numpy()[edges.i]
        edges.rename(columns={"i": "cell_i", "j": "cell_j"}).to_csv(OUT / f"edges_{graph}.csv", index=False)
        edges_per_graph[graph] = edges

        mean, lowest, at_bottleneck = cell_curvature(edges, adata.n_obs)
        cells.append(
            pd.DataFrame(
                {
                    "cell": np.arange(adata.n_obs),
                    "sample": adata.obs["sample"].to_numpy(),
                    "split": adata.obs["split"].to_numpy(),
                    "graph": graph,
                    "neighbours": degree,
                    "received": sp.csr_matrix(adj).getnnz(axis=0),
                    "curvature_mean": mean,
                    "curvature_min": lowest,
                    "at_bottleneck": at_bottleneck,
                }
            )
        )
        print(
            f"{graph:9s} {len(edges):7d} edges, curvature from {edges.curvature.min():+.2f} to {edges.curvature.max():+.2f}"
        )
    cells = pd.concat(cells, ignore_index=True)
    cells.to_csv(OUT / "cells_curvature.csv", index=False)
    return edges_per_graph, cells


def part2(edges_per_graph, cells):
    """Summary table per graph, histograms of the curvature, and curvature against edge length.
    Compare the distribution: a summary table, a histogram of edge curvature, and a plot of curvature
    against edge length.
    """
    rows = {}
    for graph, e in edges_per_graph.items():
        c = cells[cells.graph == graph]
        rows[graph] = {
            "edges": len(e),
            "neighbours per cell": c.neighbours.mean(),
            "cells receiving no messages": int((c.received == 0).sum()),
            "mean curvature": e.curvature.mean(),
            "median curvature": e.curvature.median(),
            "1% quantile": e.curvature.quantile(0.01),
            "lowest curvature": e.curvature.min(),
            "% edges < 0": 100 * (e.curvature < 0).mean(),
            "% edges < -1": 100 * (e.curvature < -1).mean(),
            "% edges bottleneck (Thm 4)": 100 * e.bottleneck.mean(),
            "% cells at a bottleneck": 100 * c.at_bottleneck.mean(),
            "Spearman(curvature, length)": spearmanr(e.curvature, e.length)[0],
            "median length": e.length.median(),
            "median length of edges < -1": e.length[e.curvature < -1].median(),
        }
    summary = pd.DataFrame(rows).T
    summary.to_csv(OUT / "curvature_summary.csv")
    print("\nCurvature per graph\n" + summary.T.to_string(float_format=fmt))

    fig, (ax_hist, ax_len) = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
    bins = np.arange(-2, 2.0001, 0.1)
    top = 0
    for graph, e in edges_per_graph.items():
        label = f"{graph} ({int(e.bottleneck.sum())} bottleneck edges)"
        density, *_ = ax_hist.hist(
            e.curvature, bins=bins, density=True, histtype="step", lw=2, color=GRAPH_COLOUR[graph], label=label
        )
        top = max(top, density.max())
    ax_hist.set_ylim(0, 1.3 * top)
    for x, text, side in [(0, "square grid", "left"), (-1 / 3, "even triangulation", "right")]:
        ax_hist.axvline(x, color="#52514e", lw=0.8, zorder=0)
        ax_hist.annotate(text, (x, 1), xycoords=("data", "axes fraction"), xytext=(3 if side == "left" else -3, -2),
                         textcoords="offset points", ha=side, va="top", fontsize=8, color="#52514e")  # fmt: skip
    ax_hist.set(xlim=(-2, 2), xlabel="Balanced Forman curvature of an edge", ylabel="density of edges")
    ax_hist.set_title("Curvature of the edges", loc="left")
    ax_hist.legend(loc="upper right", bbox_to_anchor=(1, 0.9), fontsize=8)

    for graph, e in edges_per_graph.items():
        lo, hi = np.percentile(e.length, [0.5, 99.5])
        bin_edges = np.geomspace(max(lo, 1e-6), hi, 25)
        which = np.digitize(e.length, bin_edges)
        stats = e.groupby(which).curvature.describe(percentiles=[0.25, 0.5, 0.75])
        stats = stats[(stats.index > 0) & (stats.index < len(bin_edges)) & (stats["count"] >= 20)]
        centre = np.sqrt(bin_edges[stats.index - 1] * bin_edges[stats.index])
        ax_len.fill_between(centre, stats["25%"], stats["75%"], color=GRAPH_COLOUR[graph], alpha=0.15, lw=0)
        ax_len.plot(centre, stats["50%"], color=GRAPH_COLOUR[graph], lw=2, label=graph)
    ax_len.set_xscale("log")
    ax_len.xaxis.set_major_formatter(ScalarFormatter())
    ax_len.xaxis.set_minor_formatter(NullFormatter())
    ax_len.set_xticks([t for t in (25, 50, 100, 200, 500, 1000, 2000, 5000) if t <= ax_len.get_xlim()[1]])
    ax_len.axhline(0, color="#52514e", lw=0.8, zorder=0)
    ax_len.set(xlabel="edge length (units of obsm['spatial'])", ylabel="curvature (median, 25-75% band)")
    ax_len.set_title("Curvature against edge length", loc="left")
    ax_len.legend(loc="lower left")
    ax_len.grid(axis="y")
    fig.savefig(OUT / "curvature_distribution.png")
    plt.close(fig)
    return summary


def part3(adata, edges_per_graph, cells, sample=MAP_SAMPLE):
    """Top: every edge of one sample coloured by its curvature. Bottom: every cell coloured by its mean curvature.
    It draws the curvature on the tissue for one sample by default the test sample. The figure has a grid of 2 rows and
    3 columns, one column per graph
    """
    assert sample in set(adata.obs["sample"]), f"MAP_SAMPLE {sample!r} is not in {sorted(set(adata.obs['sample']))}"
    xy = adata.obsm["spatial"]
    fig, axes = plt.subplots(2, len(GRAPHS), figsize=(4 * len(GRAPHS) + 1, 8.6), layout="constrained")
    for col, graph in enumerate(GRAPHS):
        e = edges_per_graph[graph]
        e = e[e["sample"] == sample].sort_values("curvature", ascending=False)  # most negative drawn on top
        segments = np.stack([xy[e.i], xy[e.j]], axis=1)
        lines = LineCollection(segments, cmap=CURVATURE_CMAP, norm=CURVATURE_NORM, linewidths=0.6)
        lines.set_array(e.curvature.to_numpy())
        axes[0, col].add_collection(lines)
        n_bottleneck = int(e.bottleneck.sum())
        axes[0, col].set_title(f"{graph}: edges\n{len(e):,} edges, {n_bottleneck} bottleneck (Thm 4)", loc="left")

        c = cells[(cells.graph == graph) & (cells["sample"] == sample)].sort_values("curvature_mean", ascending=False)
        axes[1, col].scatter(*xy[c.cell].T, c=c.curvature_mean, cmap=CURVATURE_CMAP, norm=CURVATURE_NORM, s=5, lw=0)
        axes[1, col].set_title(f"{graph}: cells\nmean curvature of each cell's edges", loc="left")
        for ax in axes[:, col]:
            ax.set_xlim(xy[c.cell, 0].min() - 50, xy[c.cell, 0].max() + 50)
            ax.set_ylim(xy[c.cell, 1].max() + 50, xy[c.cell, 1].min() - 50)
            ax.set_aspect("equal")
            ax.set_axis_off()
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=CURVATURE_NORM, cmap=CURVATURE_CMAP), ax=axes, shrink=0.45)
    bar.set_label("Balanced Forman curvature\n(< 0 bottleneck-like, 0 grid, > 0 clique-like)")
    fig.suptitle(f"Sample {sample}", x=0.02, ha="left", fontsize=11)
    fig.savefig(OUT / f"curvature_tissue_{sample}.png")
    plt.close(fig)


# Part 4: Are cells in negatively curved regions predicted worse?
def masked_errors(predict, data, n_groups, rounds, seed):
    """Per cell: mean squared error over genes of its predicted expression while it is hidden, averaged over rounds."""
    x = data.x.float()
    total = torch.zeros(x.shape[0])
    generator = torch.Generator().manual_seed(seed)
    for _ in range(rounds):
        for hidden in torch.randperm(x.shape[0], generator=generator).chunk(n_groups):
            x_masked = x.clone()
            x_masked[hidden] = MASK_VALUE
            prediction = predict(x_masked, data.edge_index, hidden)
            total[hidden] += ((prediction - x[hidden]) ** 2).mean(dim=1)
    return (total / rounds).numpy()


def gnn_predictor(module):
    """The trained local GNN and its decoder, as in training (eval mode: no dropout)."""

    @torch.no_grad()
    def predict(x_masked, edge_index, hidden):
        return module.predict(module(x_masked, edge_index))[hidden]

    return predict


def neighbour_mean(x_masked, edge_index, hidden):
    """Mean expression of the visible cells that send the hidden cell a message; the sample mean if there are none."""
    source, target = edge_index
    visible = torch.ones(x_masked.shape[0], dtype=torch.bool)
    visible[hidden] = False
    keep = visible[source]
    sums = torch.zeros_like(x_masked).index_add_(0, target[keep], x_masked[source[keep]])
    counts = torch.zeros(x_masked.shape[0]).index_add_(0, target[keep], torch.ones(int(keep.sum())))
    prediction = sums / counts.clamp(min=1).unsqueeze(1)
    prediction[counts == 0] = x_masked[visible].mean(dim=0)
    return prediction[hidden]


def load_model(adata, cfg):
    """The trained LocalModel saved by train_local.py, or None if it is not there yet."""
    path = Path(cfg.model.save) / f"{get_model_filename_prefix(cfg, True, False)}model.pt"
    if not path.exists():
        return None
    LocalModel._setup_anndata(
        adata,
        prediction_task=cfg.dataset.prediction_task,
        prediction_obs=cfg.dataset.prediction_obs,
        layer_key=cfg.dataset.layer_key,
        sample_key_list=list(cfg.dataset.sample_key),
        split_key=cfg.dataset.split_key,
        view_registry=False,
    )
    return LocalModel.load(cfg.model.save, adata, cfg, local_component=True)


def partial_spearman(x, y, covariates, sample):
    """Rank correlation of x and y after removing what the covariates (ranked) and the sample explain."""
    covariates = np.column_stack(
        [np.ones(len(x))]
        + [rankdata(c) for c in covariates]
        + [pd.get_dummies(sample, drop_first=True).to_numpy(float)]
    )
    residuals = []
    for v in (rankdata(x), rankdata(y)):
        coef, *_ = np.linalg.lstsq(covariates, v, rcond=None)
        residuals.append(v - covariates @ coef)
    return np.corrcoef(*residuals)[0, 1]


def part4(adata, cells):
    """Error of every evaluation cell under every model, and how it changes with the cell's curvature."""
    layer = load_config(CONFIG.format(gnn=GNNS[0].lower(), graph=GRAPHS[0])).dataset.layer_key
    x = sp.csr_matrix(adata.layers[layer] if layer else adata.X)  # the expression the models reconstruct
    expressed = np.asarray(x.sum(axis=1)).ravel() > 0  # the all-zero cells are not scored
    scored = adata.obs["split"].isin(EVAL_SPLITS).to_numpy() & expressed
    errors = []
    for graph in GRAPHS:
        cfgs = {gnn: load_config(CONFIG.format(gnn=gnn.lower(), graph=graph)) for gnn in GNNS}
        cfg = cfgs[GNNS[0]]
        for c in cfgs.values():
            assert c.dataset.connectivity_key == f"{graph}_connectivities", f"{graph} config uses another graph"
            assert c.dataset.mask_strategy == "node", "this evaluation hides whole cells, as mask_strategy 'node'"
        n_groups = round(1 / cfg.dataset.mask_percentage)

        pyg, _ = list(prepare_a2d_dataset(cfg)(adata))  # one graph per sample, built exactly as for the models
        received = cells.loc[cells.graph == graph, "received"].to_numpy()
        samples = []
        for data in pyg:
            rows = data.obs_names.numpy().astype(int) - 1  # obs_names are "1".."n", the row in adata plus one
            # the GNN graph must be the stored graph: every cell receives as many messages as its column holds
            assert np.array_equal(np.bincount(data.edge_index[1].numpy(), minlength=len(rows)), received[rows])
            if scored[rows].any():
                samples.append((data, rows))

        predictors = {"neighbour mean": neighbour_mean}
        for gnn, c in cfgs.items():
            model = load_model(adata, c)
            if model is None:
                print(f"  {gnn} on {graph}: no trained model in {c.model.save}, skipped (run train_local.py first)")
                continue
            predictors[gnn] = gnn_predictor(model.module.cpu().eval())

        for name, predict in predictors.items():
            for data, rows in samples:
                error = masked_errors(predict, data, n_groups, N_ROUNDS, SEED)
                keep = scored[rows]
                errors.append(pd.DataFrame({"cell": rows[keep], "graph": graph, "model": name, "error": error[keep]}))
            print(f"  {graph:9s} {name:15s} scored")

    errors = pd.concat(errors, ignore_index=True).merge(cells, on=["cell", "graph"], how="left")
    # Two things besides curvature make a cell's error large: a high expression level (the error of predicting all
    # zeros, which is roughly what an untrained model does) and being different from its direct neighbours (the
    # neighbour-mean error of the same cell, under the same masks).
    errors["expression_level"] = np.asarray(x.multiply(x).mean(axis=1)).ravel()[errors.cell]
    baseline = errors.loc[errors.model == "neighbour mean", ["graph", "cell", "error"]]
    errors = errors.merge(baseline.rename(columns={"error": "neighbour_mean_error"}), on=["graph", "cell"], how="left")
    errors.to_csv(OUT / "cells_error.csv", index=False)
    rows = []
    for (graph, name), e in errors.groupby(["graph", "model"], sort=False):
        e = e[e.curvature_mean.notna()]  # cells without edges have no curvature
        low = e.curvature_mean <= e.curvature_mean.quantile(0.1)
        controls = [e.received, e.expression_level]
        rows.append(
            {
                "graph": graph,
                "model": name,
                "cells": len(e),
                "MSE": e.error.mean(),
                "rho": spearmanr(e.error, e.curvature_mean)[0],
                "partial rho": partial_spearman(e.error, e.curvature_mean, controls, e["sample"]),
                "rho beyond neighbour mean": np.nan
                if name == "neighbour mean"
                else partial_spearman(e.error, e.curvature_mean, [*controls, e.neighbour_mean_error], e["sample"]),
                "MSE lowest 10% / rest": e.error[low].mean() / e.error[~low].mean(),
            }
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT / "error_vs_curvature.csv", index=False)
    print("\nError on hidden cells vs curvature\n" + stats.to_string(index=False, float_format=fmt, na_rep="-"))

    # Per decile of the cells' curvature (deciles per graph, so all models of a graph share the same cells):
    # top    each model's mean error divided by its mean error over all cells: a flat line at 1 does not depend on
    #        curvature, whatever the model's overall level (that is in the table)
    # bottom each GNN's mean error divided by the neighbour mean's on the same cells: below 1 the GNN beats the
    #        neighbour mean; a line rising towards negative curvature is a GNN losing its advantage there
    fig, axes = plt.subplots(2, len(GRAPHS), figsize=(4.2 * len(GRAPHS), 7), sharex="col", sharey="row")
    for col, graph in enumerate(GRAPHS):
        e = errors[(errors.graph == graph) & errors.curvature_mean.notna()].copy()
        e["decile"] = pd.qcut(e.curvature_mean, 10, labels=False, duplicates="drop")
        for name, (colour, marker, style) in MODEL_STYLE.items():
            m = e[e.model == name]
            if m.empty:
                continue
            per = m.groupby("decile").agg(
                x=("curvature_mean", "median"),
                y=("error", "mean"),
                se=("error", "sem"),
                base=("neighbour_mean_error", "mean"),
            )
            style_kw = dict(color=colour, marker=marker, ms=5, ls=style, lw=1.6, label=name)
            mean = m.error.mean()
            axes[0, col].errorbar(per["x"], per["y"] / mean, yerr=per["se"] / mean, elinewidth=0.8, **style_kw)
            if name != "neighbour mean":
                axes[1, col].plot(per["x"], per["y"] / per["base"], **style_kw)
        for ax in axes[:, col]:
            ax.axhline(1, color="#52514e", lw=0.8, zorder=0)
            ax.grid(axis="y")
        axes[0, col].set_title(graph, loc="left")
        axes[1, col].set_xlabel("cell curvature (mean of its edges), deciles")
    axes[0, 0].set_ylabel("MSE in the decile / model's mean MSE")
    axes[1, 0].set_ylabel("GNN MSE / neighbour-mean MSE")
    axes[0, -1].legend(loc="upper right")
    fig.suptitle(f"Hidden-cell error by curvature ({' + '.join(EVAL_SPLITS)} samples)", x=0.02, ha="left", fontsize=11)
    fig.savefig(OUT / "error_vs_curvature.png")
    plt.close(fig)
    return stats


# Part 5: Summary
def part5(summary, stats):
    """Graph properties next to each model's error and its sensitivity to curvature."""
    wide = stats.pivot(index="graph", columns="model", values=["MSE", "rho beyond neighbour mean"])
    wide = wide.dropna(axis=1, how="all")  # the neighbour mean has no "beyond neighbour mean"
    wide.columns = [f"{model}: {value}" for value, model in wide.columns]
    keep = ["mean curvature", "% edges < -1", "% edges bottleneck (Thm 4)", "cells receiving no messages"]
    table = summary[keep].join(wide).loc[GRAPHS]
    table.to_csv(OUT / "summary.csv")
    print("\nSummary (one column per graph)\n" + table.T.to_string(float_format=fmt, na_rep="-"))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    check_against_paper()

    adata = ad.read_h5ad(DATA)
    adata.obs_names = [str(i) for i in range(1, adata.n_obs + 1)]  # what InterScale's dataset builders expect

    edges_per_graph, cells = part1(adata)
    summary = part2(edges_per_graph, cells)
    part3(adata, edges_per_graph, cells)
    stats = part4(adata, cells)
    part5(summary, stats)
    print(f"\nAll outputs are in {OUT}/")
