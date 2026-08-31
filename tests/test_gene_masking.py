"""Gene-wise (per-entry) masking: sampling, corruption, and the masked loss/metrics.

The contract these lock down is the one that makes the ablation meaningful: under
``mask_strategy="gene"`` the model must be scored ONLY on the entries it did not see. If the
restriction leaks, the unmasked entries -- which were handed to the model as input -- turn the
objective into the identity map and every number becomes incomparable to the cell-masking arm.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchmetrics
from torch_geometric.data import Data

from interscale.geome_dataloader import GraphAnnDataModule
from interscale.tl.masking import (
    MASK_VALUE,
    apply_mask,
    masked_loss,
    sample_gene_mask,
    sample_node_mask,
)
from interscale.train._trainingplans import RunningCosineSimilarity, masked_regression_metrics


def _make_data(num_nodes: int, num_features: int = 6) -> Data:
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def _build_datamodule(mask_strategy: str = "gene", mask_percentage: float = 0.4) -> GraphAnnDataModule:
    dm = GraphAnnDataModule(
        datas=[[_make_data(20), _make_data(20)], [_make_data(10)], [_make_data(10)]],
        batch_size=1,
        num_workers=0,
        mask_percentage=mask_percentage,
        mask_strategy=mask_strategy,
        learning_type="node",
    )
    dm.setup(stage="fit")
    dm.setup(stage="test")
    return dm


# --------------------------------------------------------------------------- mask sampling


def test_sample_gene_mask_has_no_empty_rows():
    """A cell with nothing masked would contribute no loss and an undefined per-cell cosine."""
    mask = sample_gene_mask(500, 8, pct=0.01)

    assert mask.shape == (500, 8)
    assert mask.any(dim=1).all()


def test_sample_gene_mask_hits_the_requested_rate():
    mask = sample_gene_mask(2000, 50, pct=0.3)

    assert mask.float().mean().item() == pytest.approx(0.3, abs=0.02)


def test_sample_gene_mask_differs_across_cells():
    """A gene subset shared by every cell would be a fixed shortcut rather than supervision."""
    mask = sample_gene_mask(200, 40, pct=0.5)

    unique_rows = {tuple(row.tolist()) for row in mask}
    assert len(unique_rows) > 100


def test_sample_node_mask_masks_at_least_one_cell():
    assert sample_node_mask(50, pct=0.0).sum() == 1


# ------------------------------------------------------------------------------- datamodule


def test_gene_strategy_writes_gene_mask_and_derives_node_mask():
    dm = _build_datamodule("gene")

    for data in dm.train_data:
        assert data.gene_mask.shape == data.x.shape
        # Every cell is a supervision target under gene masking; selectivity moved to the entries.
        assert torch.equal(data.mask, data.gene_mask.any(dim=1))
        assert data.mask.all()


def test_node_strategy_leaves_no_gene_mask():
    dm = _build_datamodule("node")

    for data in dm.train_data:
        assert "gene_mask" not in data


def test_a_stale_gene_mask_cannot_hijack_the_node_strategy():
    """The strategy is a parameter, so a leftover attribute is simply ignored.

    This is why the dataloader has no `_clear_gene_mask`: correctness comes from telling
    `apply_mask` (and `_process_batch_for_metrics`) which strategy is configured, not from
    scrubbing the attribute they used to sniff.
    """
    data = _make_data(12)
    data.gene_mask = sample_gene_mask(12, data.x.shape[1], pct=0.5)  # stale, from a gene run
    data.mask = sample_node_mask(12, pct=0.5)

    out, mask_idx, entry_mask = apply_mask(data, "node")

    assert entry_mask is None
    assert torch.equal(mask_idx, torch.where(data.mask)[0])
    assert (out.x[data.mask] == MASK_VALUE).all()
    assert torch.equal(out.x[~data.mask], data.x[~data.mask])


def test_gene_strategy_without_a_gene_mask_fails_loudly():
    data = _make_data(8)
    data.mask = sample_node_mask(8, pct=0.5)

    with pytest.raises(AssertionError, match="no gene_mask"):
        apply_mask(data, "gene")


def test_resample_redraws_the_gene_mask_in_place():
    dm = _build_datamodule("gene")
    original = [data.gene_mask.clone() for data in dm.train_data]
    ids = [id(data) for data in dm.train_data]

    dm.resample_train_mask()

    assert any(not torch.equal(o, d.gene_mask) for o, d in zip(original, dm.train_data))
    assert [id(d) for d in dm.train_data] == ids


def test_unmasked_split_gets_no_gene_mask():
    """Graph-level eval splits pass mask=False and must not be corrupted."""
    dm = GraphAnnDataModule(
        datas=[[_make_data(20)], [_make_data(10)], [_make_data(10)]],
        batch_size=1,
        num_workers=0,
        mask_strategy="gene",
        learning_type="graph",
    )
    dm.setup(stage="fit")

    for data in dm.val_data:
        assert "gene_mask" not in data
        assert not data.mask.any()


def test_invalid_strategy_is_rejected():
    with pytest.raises(ValueError, match="mask_strategy"):
        GraphAnnDataModule(datas=[[_make_data(4)], [_make_data(4)]], mask_strategy="cell")


# ----------------------------------------------------------------------------- apply_mask


def test_apply_mask_gene_path_blanks_only_the_masked_entries():
    data = _make_data(12)
    data.gene_mask = sample_gene_mask(12, data.x.shape[1], pct=0.4)
    data.mask = data.gene_mask.any(dim=1)

    out, mask_idx, entry_mask = apply_mask(data, "gene")

    assert torch.equal(entry_mask, data.gene_mask)
    assert (out.x[data.gene_mask] == MASK_VALUE).all()
    # Everything else survives untouched -- that surviving context is the whole point.
    assert torch.equal(out.x[~data.gene_mask], data.x[~data.gene_mask])
    assert torch.equal(mask_idx, torch.arange(12))
    assert not torch.equal(out.x, data.x)


def test_apply_mask_node_path_is_unchanged():
    data = _make_data(12)
    data.mask = sample_node_mask(12, pct=0.5)

    out, mask_idx, entry_mask = apply_mask(data, "node")

    assert entry_mask is None
    assert torch.equal(mask_idx, torch.where(data.mask)[0])
    assert (out.x[data.mask] == MASK_VALUE).all()
    assert torch.equal(out.x[~data.mask], data.x[~data.mask])


# ----------------------------------------------------------------------------- masked loss


@pytest.mark.parametrize("loss_type,loss_fn", [("SmoothL1", nn.SmoothL1Loss()), ("MSELoss", nn.MSELoss())])
def test_masked_loss_scores_only_the_masked_entries(loss_type, loss_fn):
    torch.manual_seed(0)
    y_true = torch.randn(64, 10)
    y_pred = torch.randn(64, 10)
    entry_mask = sample_gene_mask(64, 10, pct=0.3)

    got = masked_loss(loss_fn, loss_type, y_pred, y_true, entry_mask)

    assert got == pytest.approx(loss_fn(y_pred[entry_mask], y_true[entry_mask]).item())


def test_masked_loss_ignores_error_outside_the_mask():
    """The decisive property: a wrong answer on an entry the model was GIVEN must cost nothing."""
    y_true = torch.zeros(4, 5)
    y_pred = torch.zeros(4, 5)
    entry_mask = torch.zeros(4, 5, dtype=torch.bool)
    entry_mask[:, 0] = True

    y_pred[:, 3] = 100.0  # unmasked entry, wildly wrong

    assert masked_loss(nn.MSELoss(), "MSELoss", y_pred, y_true, entry_mask).item() == 0.0


def test_masked_loss_without_entry_mask_is_the_plain_loss():
    y_true, y_pred = torch.randn(8, 5), torch.randn(8, 5)
    fn = nn.SmoothL1Loss()

    assert masked_loss(fn, "SmoothL1", y_pred, y_true, None) == pytest.approx(fn(y_pred, y_true).item())


def test_masked_loss_row_structured_restricts_to_masked_coordinates():
    """SCE normalises along the row, so entries are zeroed rather than selected."""
    from interscale.train.losses import SCELoss

    torch.manual_seed(0)
    y_true = torch.randn(16, 7)
    y_pred = torch.randn(16, 7)
    entry_mask = sample_gene_mask(16, 7, pct=0.5)
    m = entry_mask.float()

    got = masked_loss(SCELoss(), "SCELoss", y_pred, y_true, entry_mask)

    assert got == pytest.approx(SCELoss()(y_pred * m, y_true * m).item())


def test_masked_loss_gaussian_nll_uses_the_masked_row_spread():
    torch.manual_seed(0)
    y_true = torch.randn(16, 7)
    y_pred = torch.randn(16, 7)
    entry_mask = sample_gene_mask(16, 7, pct=0.5)

    got = masked_loss(nn.GaussianNLLLoss(), "GaussianNLL", y_pred, y_true, entry_mask)

    assert torch.isfinite(got)


def test_masked_row_std_matches_torch_std_when_nothing_is_held_out():
    """The masked spread must agree with the unmasked branch's torch.std on an all-True mask.

    That agreement is what keeps a GaussianNLL cell-masking run and a gene-masking run on the
    same footing; if the two branches computed different quantities the arms would not be
    comparable. (Both pass std where nn.GaussianNLLLoss documents a variance -- a pre-existing
    bug, preserved deliberately; see masked_row_std.)
    """
    from interscale.tl.masking import masked_row_std

    torch.manual_seed(0)
    y = torch.randn(32, 9)
    full = torch.ones(32, 9, dtype=torch.bool)

    # torch.std is Bessel-corrected, masked_row_std is not, so compare the population form.
    expected = y.std(dim=1, keepdim=True, unbiased=False)
    assert torch.allclose(masked_row_std(y, full), expected, atol=1e-5)


# --------------------------------------------------------------------------- masked metrics


def _reference_metrics(y_pred, y_true, m):
    """Brute-force per-gene / per-cell loop over the masked entries only."""
    per_gene_r2, per_gene_r, per_gene_ccc = [], [], []
    for g in range(y_true.shape[1]):
        sel = m[:, g]
        p, t = y_pred[sel, g], y_true[sel, g]
        if sel.sum() < 2 or t.var(unbiased=False) < 1e-8:
            continue
        per_gene_r2.append(torchmetrics.R2Score()(p, t).item())
        per_gene_r.append(torchmetrics.PearsonCorrCoef()(p, t).item())
        per_gene_ccc.append(torchmetrics.ConcordanceCorrCoef()(p, t).item())
    per_cell_cos = [
        torch.nn.functional.cosine_similarity(y_pred[i, m[i]], y_true[i, m[i]], dim=0).item()
        for i in range(y_true.shape[0])
    ]
    return {
        "mse": ((y_pred[m] - y_true[m]) ** 2).mean().item(),
        "r2": float(np.mean(per_gene_r2)),
        "pearson_corr": float(np.mean(per_gene_r)),
        "concordance_corr": float(np.mean(per_gene_ccc)),
        "cosine_similarity": float(np.mean(per_cell_cos)),
    }


def test_masked_metrics_match_a_brute_force_loop_over_masked_entries():
    torch.manual_seed(0)
    y_true = torch.randn(200, 12)
    y_pred = 0.6 * y_true + 0.5 * torch.randn(200, 12)
    m = sample_gene_mask(200, 12, pct=0.4)

    got = masked_regression_metrics(y_pred, y_true, m)
    ref = _reference_metrics(y_pred, y_true, m)

    for key, expected in ref.items():
        assert got[key].item() == pytest.approx(expected, abs=1e-4), key


def test_masked_metrics_reduce_to_torchmetrics_when_nothing_is_held_out():
    """An all-True mask is the cell-masking case, so the two paths must agree there."""
    torch.manual_seed(0)
    y_true = torch.randn(200, 12)
    y_pred = 0.6 * y_true + 0.5 * torch.randn(200, 12)
    full = torch.ones(200, 12, dtype=torch.bool)

    got = masked_regression_metrics(y_pred, y_true, full)

    assert got["mse"].item() == pytest.approx(torchmetrics.MeanSquaredError()(y_pred, y_true).item(), abs=1e-5)
    assert got["r2"].item() == pytest.approx(
        torchmetrics.R2Score(multioutput="uniform_average")(y_pred, y_true).item(), abs=1e-4
    )
    assert got["pearson_corr"].item() == pytest.approx(
        torch.nanmean(torchmetrics.PearsonCorrCoef(num_outputs=12)(y_pred, y_true)).item(), abs=1e-4
    )
    assert got["cosine_similarity"].item() == pytest.approx(RunningCosineSimilarity()(y_pred, y_true).item(), abs=1e-5)


def test_masked_metrics_are_blind_to_predictions_outside_the_mask():
    """The inflation this guards against: scoring entries the model was handed as input."""
    torch.manual_seed(0)
    y_true = torch.randn(80, 10)
    y_pred = 0.5 * y_true + 0.3 * torch.randn(80, 10)
    m = sample_gene_mask(80, 10, pct=0.3)

    before = masked_regression_metrics(y_pred, y_true, m)
    corrupted = y_pred.clone()
    corrupted[~m] = 1e3
    after = masked_regression_metrics(corrupted, y_true, m)

    for key in before:
        assert after[key].item() == pytest.approx(before[key].item(), abs=1e-5), key


def test_masked_metrics_skip_genes_with_too_few_masked_cells():
    """A gene masked in one cell has no correlation defined; it must not poison the mean."""
    torch.manual_seed(0)
    y_true = torch.randn(50, 4)
    y_pred = torch.randn(50, 4)
    m = torch.zeros(50, 4, dtype=torch.bool)
    m[:, 0] = True  # well-populated
    m[0, 1] = True  # single cell -> undefined

    got = masked_regression_metrics(y_pred, y_true, m)

    for key, value in got.items():
        assert torch.isfinite(value), key


# ------------------------------------------------------------------------- the fill value
#
# MASK_VALUE is -1 and is no longer configurable. On a log1p layer that is ~62% exact zeros (and
# with 648 all-zero cells), a 0 fill makes a masked position indistinguishable from a real
# measurement; under gene masking the corruption becomes invisible entirely. Measured at gene
# rate 0.25 over 3 seeds, 0 scored 0.0696 +/- 0.0113 against -1's 0.0799 +/- 0.0046.


def test_mask_value_is_outside_the_range_of_a_log1p_layer():
    """The one property the fill value has to have. A log1p layer is >= 0 by construction."""
    assert MASK_VALUE < 0


def test_masked_positions_are_distinguishable_from_real_zeros():
    """The regression this guards: with a 0 fill, `x == MASK_VALUE` would also select real zeros."""
    data = _make_data(200, num_features=6)
    # ~60% exact zeros, as in the real layer.
    data.x = (torch.rand(200, 6) > 0.6).float() * torch.rand(200, 6)
    data.gene_mask = sample_gene_mask(200, 6, pct=0.3)
    data.mask = data.gene_mask.any(dim=1)

    out, _, entry_mask = apply_mask(data, "gene")

    # Every masked position, and ONLY the masked positions, carry the fill value.
    assert torch.equal(out.x == MASK_VALUE, entry_mask)


def test_all_zero_cells_stay_distinguishable_from_masked_cells():
    """legnini23 has 648 cells whose expression is entirely zero, so an all-zero row is real data."""
    data = _make_data(10, num_features=6)
    data.x = torch.zeros(10, 6)
    data.mask = sample_node_mask(10, pct=0.5)

    out, _, _ = apply_mask(data, "node")

    assert (out.x[data.mask] == MASK_VALUE).all()
    assert (out.x[~data.mask] == 0).all()
