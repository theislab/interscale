"""Tests for `interscale.tl.anchors` and the figures in `interscale.pl.anchor_plots`.

The fixture is a two-slide synthetic stand-in for legnini23: `slideA` carries two compact
foci of one latent dimension plus a gene that decays slowly away from them (long range), a
gene that decays fast (local), and a flat gene; `slideB` carries none of it. That asymmetry
is what the assertions are about -- a construction that finds anchors on a slide with no
structure, or that measures a distance across slides, would pass a single-slide test.
"""

from types import SimpleNamespace

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from interscale.pl import (
    anchor_map,
    distance_profile,
    gene_maps,
    signed_distance_map,
    zone_map,
)
from interscale.tl import (
    anchor_enrichment,
    anchor_signed_distance,
    anchor_zones,
    find_anchor_cells,
    local_reach_um,
    profile_by_distance,
)
from interscale.tl.anchors import ZONE_CORE, ZONE_FAR, ZONE_NEAR

FOCI = np.array([[1500.0, 1500.0], [4500.0, 4200.0]])
GENES = ["LONG", "SHORT", "FLAT"]


@pytest.fixture(scope="module")
def anchor_adata():
    rng = np.random.default_rng(0)
    n_per = 1200
    xy = np.vstack([rng.uniform(0, 6000, size=(n_per, 2)) for _ in range(2)])
    sample = np.array(["slideA"] * n_per + ["slideB"] * n_per)
    on_a = sample == "slideA"
    n = len(xy)

    # Distance to the nearest planted focus, the ground truth every assertion is against.
    d = np.min(np.linalg.norm(xy[:, None, :] - FOCI[None], axis=2), axis=1)

    Z = rng.normal(0, 1, size=(n, 4))
    # Dimension 1 peaks at the foci on slideA only, and on its NEGATIVE pole -- a latent
    # dimension's sign is arbitrary, so sign="auto" has to find this.
    Z[:, 1] = np.where(on_a, -4.0 * np.exp(-((d / 600.0) ** 2)), 0.0) + rng.normal(0, 0.3, n)
    # Dimension 2 is offset on slideB, which a global (rather than per-sample) threshold
    # would let take every anchor in the dataset.
    Z[:, 2] = np.where(on_a, 0.0, 6.0) + rng.normal(0, 0.3, n)

    X = np.empty((n, 3))
    X[:, 0] = np.where(on_a, 3.0 * np.exp(-d / 2500.0), 0.2)  # decays past the local reach
    X[:, 1] = np.where(on_a, 2.0 * np.exp(-d / 900.0), 0.1)  # decays within it
    X[:, 2] = 1.0  # flat
    X = np.clip(X + rng.normal(0, 0.1, X.shape), 0, None)

    adata = ad.AnnData(X=X.astype(np.float32))
    adata.var_names = GENES
    adata.layers["log1p_norm"] = adata.X.copy()
    adata.obs["sample"] = pd.Categorical(sample)
    adata.obs["truth"] = pd.Categorical(np.where(d < 800, "focus", "other"))
    adata.obsm["spatial"] = xy
    adata.obsm["40_global_emb"] = Z
    adata.uns["d_to_focus"] = d
    return adata


@pytest.fixture(scope="module")
def anchored(anchor_adata):
    """(anchor, focus, dist, zone) for dimension 1 -- the planted long-range dimension."""
    anchor, focus = find_anchor_cells(anchor_adata, "40_global_emb", 1, min_cluster_size=8)
    dist = anchor_signed_distance(anchor_adata, anchor)
    zone = anchor_zones(anchor_adata, dist, near_max=400.0)
    return anchor, focus, dist, zone


# ------------------------------------------------------------------------------------------
# local_reach_um
# ------------------------------------------------------------------------------------------


def _cfg(radius, num_layers=2):
    return SimpleNamespace(
        dataset=SimpleNamespace(spatial_neigbors_kwargs=SimpleNamespace(radius=radius)),
        model=SimpleNamespace(local_component=SimpleNamespace(parameters=SimpleNamespace(num_layers=num_layers))),
    )


def test_local_reach_is_hops_times_radius():
    assert local_reach_um(_cfg(200, 2)) == 400.0


def test_local_reach_raises_for_knn_graph():
    # radius None means a neighbour-count graph, which has no reach in um to report.
    with pytest.raises(ValueError, match="cannot be expressed in um"):
        local_reach_um(_cfg(None))


# ------------------------------------------------------------------------------------------
# find_anchor_cells
# ------------------------------------------------------------------------------------------


def test_finds_the_planted_foci_on_the_right_slide(anchor_adata, anchored):
    anchor, focus, _, _ = anchored
    sample = anchor_adata.obs["sample"].to_numpy()
    assert anchor.sum() > 0
    assert anchor[sample == "slideB"].sum() == 0, "anchored on a slide with no structure"
    assert len(set(focus[anchor])) == len(FOCI)
    # anchors sit at the foci, not scattered over the slide
    d = anchor_adata.uns["d_to_focus"]
    assert d[anchor].mean() < 0.5 * d[sample == "slideA"].mean()


def test_auto_sign_takes_the_informative_pole(anchor_adata):
    anchor, _ = find_anchor_cells(anchor_adata, "40_global_emb", 1, min_cluster_size=8, key_added="a")
    assert anchor_adata.uns["a_info"]["sign"] == -1.0
    # forcing the wrong pole must not reproduce the foci
    wrong, _ = find_anchor_cells(anchor_adata, "40_global_emb", 1, sign="+", min_cluster_size=8)
    assert not (anchor & wrong).any()


def test_threshold_is_per_sample(anchor_adata):
    # Dimension 2 is +6 on slideB and 0 on slideA. A dataset-wide quantile would put every
    # anchor on slideB; a per-sample one splits them.
    anchor, _ = find_anchor_cells(anchor_adata, "40_global_emb", 2, min_cluster_size=1)
    sample = anchor_adata.obs["sample"].to_numpy()
    assert anchor[sample == "slideA"].sum() > 0
    assert anchor[sample == "slideB"].sum() > 0


def test_tighter_quantile_gives_fewer_anchors(anchor_adata):
    loose, _ = find_anchor_cells(anchor_adata, "40_global_emb", 1, quantile=0.95, min_cluster_size=4)
    tight, _ = find_anchor_cells(anchor_adata, "40_global_emb", 1, quantile=0.99, min_cluster_size=4)
    assert 0 < tight.sum() < loose.sum()
    assert tight.sum() < 0.02 * anchor_adata.n_obs, "a threshold this tight must leave few anchors"


def test_min_cluster_size_discards_scattered_cells(anchor_adata):
    kept, _ = find_anchor_cells(anchor_adata, "40_global_emb", 3, min_cluster_size=1)
    dropped, _ = find_anchor_cells(anchor_adata, "40_global_emb", 3, min_cluster_size=25)
    # dimension 3 is pure noise: its tail has no spatial focus to survive the size filter
    assert kept.sum() > 0
    assert dropped.sum() == 0


def test_link_radius_merges_foci(anchor_adata):
    # The two planted foci are ~4000 um apart, so linking at 5000 makes them one component.
    # Restricted to slideA: at that link radius the whole of a structureless slide links up
    # too (see test_link_radius_too_large_manufactures_a_focus), which would count as a
    # second focus here for a reason that has nothing to do with merging.
    kw = {"min_cluster_size": 8, "samples": ["slideA"]}
    _, near = find_anchor_cells(anchor_adata, "40_global_emb", 1, link_radius=400.0, **kw)
    _, far = find_anchor_cells(anchor_adata, "40_global_emb", 1, link_radius=5000.0, **kw)
    assert len(set(near[near >= 0])) == 2
    assert len(set(far[far >= 0])) == 1


def test_link_radius_too_large_manufactures_a_focus(anchor_adata):
    # The spatial-coherence filter IS link_radius: raised past the slide's own extent, every
    # tail cell links to every other and the tail of a dimension with no structure comes back
    # as one big "focus". slideB has nothing planted in dimension 1, so it must stay empty at
    # the default and only appear when the radius is absurd -- which is why the default is
    # tied to the local component's reach rather than picked for how many foci it yields.
    sample = anchor_adata.obs["sample"].to_numpy()
    default, _ = find_anchor_cells(anchor_adata, "40_global_emb", 1, min_cluster_size=8)
    absurd, _ = find_anchor_cells(anchor_adata, "40_global_emb", 1, link_radius=20_000.0, min_cluster_size=8)
    assert default[sample == "slideB"].sum() == 0
    assert absurd[sample == "slideB"].sum() > 0


def test_key_added_records_what_was_thresholded(anchor_adata):
    find_anchor_cells(anchor_adata, "40_global_emb", 1, min_cluster_size=8, key_added="anchor1")
    info = anchor_adata.uns["anchor1_info"]
    assert anchor_adata.obs["anchor1"].dtype == bool
    assert set(anchor_adata.obs.loc[~anchor_adata.obs["anchor1"], "anchor1_focus"]) == {-1}
    assert info["dim"] == 1 and info["n_foci"] == 2
    assert set(info["thresholds"]) == {"slideA", "slideB"}


def test_samples_restricts_without_moving_the_threshold(anchor_adata):
    both, _ = find_anchor_cells(anchor_adata, "40_global_emb", 1, min_cluster_size=8)
    one, _ = find_anchor_cells(anchor_adata, "40_global_emb", 1, min_cluster_size=8, samples=["slideA"])
    assert np.array_equal(both, one)  # slideB had no anchor to lose


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"quantile": 1.0}, "quantile must be"),
        ({"sign": "up"}, "sign must be"),
        ({"dim": 99}, "out of range"),
        ({"emb_key": "nope"}, "not found in adata.obsm"),
    ],
)
def test_find_anchor_cells_rejects_bad_arguments(anchor_adata, kwargs, match):
    call = {"emb_key": "40_global_emb", "dim": 1, **kwargs}
    with pytest.raises((ValueError, KeyError), match=match):
        find_anchor_cells(anchor_adata, **call)


# ------------------------------------------------------------------------------------------
# anchor_signed_distance / anchor_zones
# ------------------------------------------------------------------------------------------


def test_signed_distance_is_negative_inside_and_nan_without_anchors(anchor_adata, anchored):
    anchor, _, dist, _ = anchored
    sample = anchor_adata.obs["sample"].to_numpy()
    assert (dist[anchor] <= 0).all()
    outside = np.isfinite(dist) & ~anchor
    assert (dist[outside] > 0).all()
    # slideB holds no focus, so there is nothing to measure a distance from -- a result, not
    # an error, and it must not silently borrow slideA's anchors.
    assert np.isnan(dist[sample == "slideB"]).all()


def test_signed_distance_never_crosses_samples(anchor_adata, anchored):
    anchor, _, dist, _ = anchored
    xy = anchor_adata.obsm["spatial"]
    sample = anchor_adata.obs["sample"].to_numpy()
    a_rows = np.where((sample == "slideA") & ~anchor)[0]
    inside = np.where(anchor)[0]
    within = np.min(np.linalg.norm(xy[a_rows][:, None] - xy[inside][None], axis=2), axis=1)
    assert np.allclose(dist[a_rows], within)


def test_signed_distance_rejects_wrong_length(anchor_adata):
    with pytest.raises(ValueError, match="must have shape"):
        anchor_signed_distance(anchor_adata, np.zeros(7, dtype=bool))


def test_zones_split_at_zero_and_at_near_max(anchor_adata):
    dist = np.full(anchor_adata.n_obs, np.nan)
    dist[:5] = [-10.0, 0.0, 1.0, 400.0, 400.1]
    zone = anchor_zones(anchor_adata, dist, near_max=400.0)
    assert list(zone[:5]) == [ZONE_CORE, ZONE_CORE, ZONE_NEAR, ZONE_NEAR, ZONE_FAR]
    assert pd.isna(zone[5])
    assert zone.ordered and list(zone.categories) == [ZONE_CORE, ZONE_NEAR, ZONE_FAR]


# ------------------------------------------------------------------------------------------
# profile_by_distance
# ------------------------------------------------------------------------------------------


def test_profile_separates_long_from_short_range(anchor_adata, anchored):
    _, _, dist, _ = anchored
    prof = profile_by_distance(anchor_adata, GENES, dist, n_bins=12, min_cells=15)
    piv = prof.pivot(index="center", columns="gene", values="mean").sort_index()
    beyond = piv.index > 400.0  # past the local component's reach

    # SHORT is spent by the time it leaves the local reach; LONG still falls beyond it.
    long_fall = piv.loc[beyond, "LONG"].iloc[0] - piv.loc[beyond, "LONG"].iloc[-1]
    short_fall = piv.loc[beyond, "SHORT"].iloc[0] - piv.loc[beyond, "SHORT"].iloc[-1]
    assert long_fall > short_fall
    assert piv["FLAT"].std() < 0.1 * piv["LONG"].std()
    assert (prof["sem"] >= 0).all() and (prof["n"] >= 15).all()


def test_profile_center_is_where_the_cells_are(anchor_adata, anchored):
    _, _, dist, _ = anchored
    prof = profile_by_distance(anchor_adata, ["LONG"], dist, n_bins=8)
    assert ((prof["center"] >= prof["lo"]) & (prof["center"] <= prof["hi"])).all()


def test_profile_mask_and_binning(anchor_adata, anchored):
    _, _, dist, _ = anchored
    sample = anchor_adata.obs["sample"].to_numpy()
    q = profile_by_distance(anchor_adata, ["LONG"], dist, mask=sample == "slideA", n_bins=10)
    e = profile_by_distance(anchor_adata, ["LONG"], dist, mask=sample == "slideA", n_bins=10, binning="equal")
    # quantile bins hold a comparable number of cells; equal-width ones do not
    assert q["n"].std() < e["n"].std()
    # slideB is all-nan, so masking it out cannot change the profile
    both = profile_by_distance(anchor_adata, ["LONG"], dist, n_bins=10)
    assert np.allclose(q["mean"], both["mean"])


def test_profile_returns_empty_frame_when_nothing_is_selected(anchor_adata, anchored):
    _, _, dist, _ = anchored
    out = profile_by_distance(anchor_adata, ["LONG"], dist, mask=np.zeros(anchor_adata.n_obs, bool))
    assert out.empty and "mean" in out.columns


def test_profile_rejects_unknown_genes_and_binning(anchor_adata, anchored):
    _, _, dist, _ = anchored
    with pytest.raises(KeyError, match="not in adata.var_names"):
        profile_by_distance(anchor_adata, ["NOPE"], dist)
    with pytest.raises(ValueError, match="binning must be"):
        profile_by_distance(anchor_adata, ["LONG"], dist, binning="log")


# ------------------------------------------------------------------------------------------
# anchor_enrichment
# ------------------------------------------------------------------------------------------


def test_enrichment_finds_what_the_foci_are_made_of(anchor_adata, anchored):
    anchor, _, _, _ = anchored
    enr = anchor_enrichment(anchor_adata, anchor, "truth").set_index(["sample", "category"])
    assert list(enr.index.get_level_values("sample").unique()) == ["slideA"]  # slideB has none
    assert enr.loc[("slideA", "focus"), "log2_enrichment"] > 2
    # a category with no cells inside the foci is -inf, not nan or 0
    assert enr.loc[("slideA", "other"), "log2_enrichment"] == -np.inf


# ------------------------------------------------------------------------------------------
# plots
# ------------------------------------------------------------------------------------------

SLIDES = ["slideA", "slideB"]


def test_anchor_map_draws_one_panel_per_sample(anchor_adata, anchored):
    anchor, focus, _, _ = anchored
    fig, axs = anchor_map(anchor_adata, anchor, focus=focus, samples=SLIDES, show=False)
    assert len(axs) == 2
    # outside + anchors on slideA, outside only on the slide with no focus
    assert len(axs[0].collections) == 2
    assert axs[1].collections[1].get_offsets().shape[0] == 0
    plt.close(fig)


def test_signed_distance_map_draws_the_anchorless_slide(anchor_adata, anchored):
    _, _, dist, _ = anchored
    fig, axs = signed_distance_map(anchor_adata, dist, samples=SLIDES, show=False)
    # an all-nan slide is drawn in the na colour rather than left empty
    assert axs[1].collections and axs[1].collections[0].get_offsets().shape[0] > 0
    plt.close(fig)


def test_signed_distance_map_survives_an_all_positive_distance(anchor_adata, anchored):
    # TwoSlopeNorm needs vmin < 0 < vmax; a slide with no interior must not raise.
    _, _, dist, _ = anchored
    d = np.where(np.isfinite(dist), np.abs(dist), np.nan)
    fig, _ = signed_distance_map(anchor_adata, d, samples=["slideA"], show=False)
    plt.close(fig)


def test_zone_and_gene_maps_accept_obs_keys(anchor_adata, anchored):
    anchor, _, dist, zone = anchored
    anchor_adata.obs["zone"] = zone
    anchor_adata.obs["is_anchor"] = anchor
    fig, axs = zone_map(anchor_adata, "zone", samples=SLIDES, show=False)
    plt.close(fig)
    fig, axs = gene_maps(anchor_adata, ["LONG", "SHORT"], samples=SLIDES, anchor="is_anchor", show=False)
    assert axs.shape == (2, 2)
    plt.close(fig)


def test_gene_maps_rejects_unknown_genes(anchor_adata):
    with pytest.raises(KeyError, match="not in adata.var_names"):
        gene_maps(anchor_adata, ["NOPE"], samples=SLIDES, show=False)


def test_distance_profile_marks_the_local_reach(anchor_adata, anchored):
    _, _, dist, _ = anchored
    prof = profile_by_distance(anchor_adata, GENES, dist, n_bins=12)
    fig, axs = distance_profile(prof, local_reach=400.0, show=False)
    xs = sorted(line.get_xdata()[0] for line in axs[0].lines if len(set(line.get_xdata())) == 1)
    assert 0.0 in xs and 400.0 in xs
    plt.close(fig)

    fig, axs = distance_profile(prof, local_reach=400.0, one_panel_per_gene=True, show=False)
    assert len(axs) == len(GENES)
    plt.close(fig)


def test_distance_profile_rejects_empty_and_unknown(anchor_adata, anchored):
    _, _, dist, _ = anchored
    prof = profile_by_distance(anchor_adata, ["LONG"], dist, n_bins=8)
    with pytest.raises(ValueError, match="prof is empty"):
        distance_profile(prof.iloc[:0], show=False)
    with pytest.raises(ValueError, match="genes not present"):
        distance_profile(prof, genes=["NOPE"], show=False)


def test_plots_reject_a_missing_sample(anchor_adata, anchored):
    anchor, _, _, _ = anchored
    with pytest.raises(KeyError, match="no group"):
        anchor_map(anchor_adata, anchor, samples=["slideZ"], show=False)
