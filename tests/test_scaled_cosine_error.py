"""GraphMAE's scaled cosine error (SCE) as a loss and as a metric, restricted to masked cells.

Hou et al., KDD 2022, eq. 2: the mean over the masked nodes of ``(1 - cos(x_i, z_i)) ** gamma``,
``gamma >= 1``. Pinned here: the definition; ``optim.sce_gamma`` reaching both the loss and the
metrics; the loss covering the masked cells only (cell masking) or each cell's masked genes
(gene masking); and the masked metrics averaging over masked cells only -- they used to count every
unmasked cell as a cosine of 0.
"""

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from interscale.config import load_config
from interscale.tl.masking import masked_loss, sample_gene_mask
from interscale.train._trainingplans import RunningScaledCosineError, TrainingPlan, masked_regression_metrics
from interscale.train.losses import SCELoss, scaled_cosine_error


def _pair(n=60, g=20, noise=0.3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    y_true = torch.rand(n, g, generator=gen) + 0.2
    y_pred = y_true + noise * torch.randn(n, g, generator=gen)
    return y_pred, y_true


def _cell_mask(n, g, rate=0.3, seed=1):
    """The entry mask cell masking produces: each masked cell's row all True, the rest all False."""
    cells = torch.rand(n, generator=torch.Generator().manual_seed(seed)) < rate
    return cells, cells[:, None].expand(-1, g)


# --------------------------------------------------------------------------- the definition


def test_scaled_cosine_error_is_graphmaes_definition():
    y_pred, y_true = _pair()
    cos = (y_pred * y_true).sum(1) / (y_pred.norm(dim=1) * y_true.norm(dim=1))

    for gamma in (1.0, 2.0, 3.0):
        assert torch.allclose(scaled_cosine_error(y_pred, y_true, gamma), (1 - cos) ** gamma, atol=1e-6)


def test_gamma_one_is_the_plain_cosine_error():
    y_pred, y_true = _pair()
    cos = nn.functional.cosine_similarity(y_pred, y_true, dim=1)

    assert torch.allclose(scaled_cosine_error(y_pred, y_true, 1.0), 1 - cos, atol=1e-6)


def test_a_larger_gamma_shifts_the_weight_to_the_hard_cells():
    easy, hard = torch.tensor([[1.0, 0.1]]), torch.tensor([[1.0, 1.5]])
    target = torch.tensor([[1.0, 0.0]])

    shares = []
    for gamma in (1.0, 3.0):
        e, h = scaled_cosine_error(easy, target, gamma), scaled_cosine_error(hard, target, gamma)
        shares.append((h / (e + h)).item())

    assert shares[1] > shares[0], "the hard cell's share of the error grows with gamma"


def test_identical_rows_give_zero_even_for_a_non_integer_gamma():
    """Rounding can put the cosine of identical rows above 1; a negative base to 1.5 is NaN."""
    y = torch.rand(50, 30) * 1e3

    out = scaled_cosine_error(y, y.clone(), 1.5)

    assert torch.isfinite(out).all() and torch.allclose(out, torch.zeros(50), atol=1e-6)


def test_the_loss_is_the_mean_over_the_rows_it_is_given_and_rejects_gamma_below_one():
    y_pred, y_true = _pair()

    assert SCELoss(gamma=2.0)(y_pred, y_true) == pytest.approx(scaled_cosine_error(y_pred, y_true, 2.0).mean().item())
    with pytest.raises(ValueError, match="gamma >= 1"):
        SCELoss(gamma=0.5)


def test_the_running_metric_is_the_mean_over_every_cell_seen():
    (p1, t1), (p2, t2) = _pair(seed=0), _pair(n=25, seed=3)
    metric = RunningScaledCosineError(gamma=2.0)
    metric.update(p1, t1)
    metric.update(p2, t2)

    expected = scaled_cosine_error(torch.cat([p1, p2]), torch.cat([t1, t2]), 2.0).mean()
    assert metric.compute().item() == pytest.approx(expected.item(), abs=1e-6)


# --------------------------------------------------------------------------- masked cells only


def test_under_cell_masking_the_loss_is_the_mean_over_the_masked_cells_only():
    """GraphMAE's average over the masked nodes: unmasked cells neither count nor matter."""
    y_pred, y_true = _pair()
    cells, entry_mask = _cell_mask(*y_pred.shape)
    loss_fn = SCELoss(gamma=2.0)

    got = masked_loss(loss_fn, "SCELoss", y_pred, y_true, entry_mask)
    corrupted = y_pred.clone()
    corrupted[~cells] = -5.0

    assert got.item() == pytest.approx(scaled_cosine_error(y_pred[cells], y_true[cells], 2.0).mean().item())
    assert masked_loss(loss_fn, "SCELoss", corrupted, y_true, entry_mask).item() == pytest.approx(got.item())


def test_under_gene_masking_the_loss_uses_each_cells_masked_genes_and_skips_cells_without_any():
    """A cell without a masked gene would otherwise enter the mean as an all-zero row: error 1."""
    y_pred, y_true = _pair(n=40, g=6)
    entry_mask = sample_gene_mask(40, 6, pct=0.3)
    entry_mask[:5] = False  # five cells with nothing masked
    assert not entry_mask.all(dim=1).all() and entry_mask.any(dim=1).any()

    got = masked_loss(SCELoss(gamma=3.0), "SCELoss", y_pred, y_true, entry_mask)

    rows = [i for i in range(40) if entry_mask[i].any()]
    expected = torch.stack(
        [scaled_cosine_error(y_pred[i, entry_mask[i]][None], y_true[i, entry_mask[i]][None], 3.0)[0] for i in rows]
    ).mean()
    assert got.item() == pytest.approx(expected.item(), abs=1e-6)


def test_the_masked_cosine_metrics_average_over_masked_cells_only():
    """They used to average over every cell, so under cell masking the unmasked cells entered as a
    cosine of 0: a near-perfect reconstruction reported ~0.35 at a 30% masking rate."""
    y_pred, y_true = _pair(n=100, g=40, noise=0.05)
    cells, entry_mask = _cell_mask(100, 40)

    got = masked_regression_metrics(y_pred, y_true, entry_mask, sce_gamma=2.0)

    cos = nn.functional.cosine_similarity(y_pred[cells], y_true[cells], dim=1).mean()
    assert got["cosine_similarity"].item() == pytest.approx(cos.item(), abs=1e-6)
    expected = scaled_cosine_error(y_pred[cells], y_true[cells], 2.0).mean()
    assert got["scaled_cosine_error"].item() == pytest.approx(expected.item(), abs=1e-6)


# --------------------------------------------------------------------------- through the config and plan


class _Stub(nn.Module):
    n_output = 20


def _plan(loss="SCELoss", gamma=2.0):
    return TrainingPlan(
        _Stub(), "regression", "node", loss, "gene", batch_size=2, sce_gamma=gamma, lr_scheduler="CosineWarmupScheduler"
    )


@pytest.mark.parametrize("loss", ["SCELoss", "SCE_EntropyATT_Loss"])
def test_the_configured_gamma_reaches_the_loss(loss):
    assert _plan(loss, gamma=2.0).loss.gamma == 2.0


def test_the_plan_logs_sce_for_all_and_for_masked_cells_and_the_masked_one_is_the_loss():
    """Same cells, same gamma: the logged masked SCE is the number the model was trained on."""
    plan = _plan("SCELoss", gamma=2.0)
    y_pred, y_true = _pair()
    cells, entry_mask = _cell_mask(*y_pred.shape)

    loss, metrics = plan._regression_metrics(y_pred, y_true, "val", plan.valid_metrics, entry_mask=entry_mask)

    assert metrics["val_masked_scaled_cosine_error"].item() == pytest.approx(loss.item(), abs=1e-6)
    every_cell = scaled_cosine_error(y_pred, y_true, 2.0).mean()
    assert metrics["val_scaled_cosine_error"].item() == pytest.approx(every_cell.item(), abs=1e-6)


def test_sce_is_tracked_whatever_the_loss():
    plan = _plan("MSELoss", gamma=1.0)
    y_pred, y_true = _pair()

    _, metrics = plan._regression_metrics(y_pred, y_true, "val", plan.valid_metrics)

    cos = nn.functional.cosine_similarity(y_pred, y_true, dim=1).mean()
    assert metrics["val_scaled_cosine_error"].item() == pytest.approx(1 - cos.item(), abs=1e-6)


def test_the_config_keeps_gamma_three_by_default_and_rejects_below_one():
    assert load_config().optim.sce_gamma == 3.0
    assert load_config(overrides=["optim.sce_gamma", 2]).optim.sce_gamma == 2.0
    with pytest.raises(ValueError, match="sce_gamma"):
        load_config(overrides=["optim.sce_gamma", 0.5])


@pytest.mark.parametrize(
    ("monitor", "mode"),
    [
        ("val_masked_scaled_cosine_error", "min"),
        ("val_scaled_cosine_error", "min"),
        ("val_mse", "min"),
        ("val_loss", "min"),
        ("val_r2", "max"),
        ("val_masked_cosine_similarity", "max"),
    ],
)
def test_early_stopping_minimises_errors(monitor, mode):
    """Monitoring the SCE must not maximise it: the direction used to follow a `loss` suffix only."""
    from types import SimpleNamespace

    from interscale.train._training import NodeMaskingTrainingPlan

    owner = SimpleNamespace(_cfg=SimpleNamespace(optim=SimpleNamespace(monitor=monitor)), prediction_task="regression")

    assert NodeMaskingTrainingPlan._resolve_monitor(owner) == (monitor, mode)
