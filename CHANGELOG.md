# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.1.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- The scaled cosine error of GraphMAE (Hou et al., KDD 2022, eq. 2), `(1 - cos(pred, true)) ** gamma`
  per cell, is logged for every regression run whatever its loss, as `{train,val,test}_scaled_cosine_error`
  over all cells and `..._masked_scaled_cosine_error` over the masked ones. With `SCELoss` as the
  criterion the masked one is the training loss itself. The exponent is the new
  `optim.sce_gamma` (default 3.0, the value `SCELoss` always used; `>= 1`, checked at config
  load), which `SCELoss` and `SCE_EntropyATT_Loss` now read too. `gamma = 1` is the plain cosine
  error, `1 - cosine_similarity`. The per-cell function is `train.losses.scaled_cosine_error`.

- Anchor-point / distance-zone analysis of a latent dimension (`interscale.tl.anchors`,
  plotted by `interscale.pl.anchor_plots`): `find_anchor_cells` localises the foci a dimension
  is reading (per-sample tail quantile, then a spatial-coherence filter),
  `anchor_signed_distance` gives every cell a signed distance to the anchor border,
  `anchor_zones` cuts that into `anchor core` / `near` / `far` at the local component's reach
  (`local_reach_um` = `num_layers` x `radius`), and `profile_by_distance` /
  `anchor_enrichment` profile expression and composition against it. This is the evidence a
  *global* dimension carries a long-range interaction rather than a neighbourhood effect:
  every distance plot marks the local reach, past which no cell's neighbourhood extends.

- Gene-wise (per-entry) masking for the node-level reconstruction task, alongside the existing
  whole-cell masking, selected by `dataset.mask_strategy` (`"node"` | `"gene"`). Under `"gene"`
  a Bernoulli subset of `(cell, gene)` entries is blanked in every cell and the loss and every
  regression metric are restricted to those entries. Registered as the `node_reg_genemask` task
  for `legnini23` and `melton25`, with `config_files/sweeps/mask_granularity_legnini.yaml`
  running the granularity/rate ablation.

### Changed

- `SCELoss` and `SCE_EntropyATT_Loss` take `gamma` (the paper's name) instead of `alpha`, and
  validate it is `>= 1`. `1 - cos` is clamped at 0 before the power, so a non-integer `gamma` no
  longer turns two identical rows into NaN.
- `tl.masking.MASK_VALUE` is now **-1**, was 0. Masked positions must be distinguishable from
  real measurements: ~62% of legnini23's `log1p_norm` entries are already exactly 0 and 648 of
  its cells are all-zero, so a zero fill made the corruption invisible under gene masking.
  Measured at gene rate 0.25 over 3 seeds, fill 0 scored `val_concordance_corr` 0.0696 ±0.0113
  against -1's 0.0799 ±0.0046 — better on every seed, with 2.5x less run-to-run spread. A
  configurable fill (`zero`/`sentinel`/`learned`, including a trainable per-gene `[MASK]` token)
  was trialled and removed: the learnable token tied with the fixed -1 to three decimals, so the
  option carried no information. **This also changes the cell-masking path**, whose published
  runs used 0.
- `dataset.pct_mask_nodes` and `dataset.pct_mask_genes` are replaced by a single
  `dataset.mask_percentage` (and `GraphAnnDataModule`/`BaseModule` take one `mask_percentage`
  argument). `mask_strategy` already says whether a masked unit is a cell or an entry, so a
  second rate was always the inert half of a pair — and setting the wrong one silently produced
  a run with no masking. `GraphAnnDataModule.mask_rate` and `BaseModule.mask_rate` are gone with
  it. **Every config using `pct_mask_nodes` must be renamed**, including saved wandb sweeps.
- `tl.masking.apply_mask` takes `mask_strategy` explicitly instead of inferring it from whether
  the batch carries a `gene_mask`, and `GlobalModule._process_batch_for_metrics` gates on the
  strategy the same way. A stale `gene_mask` can no longer hijack a cell-masking run, so
  `GraphAnnDataModule._clear_gene_mask` is removed.
- `tl.masking.apply_mask` and `BaseModule._common_step_masking` now return a third value, the
  entry mask; every `_common_step` returns a sixth value carrying it through to the training plan.
  `GlobalModule._process_batch_for_metrics` returns it as a third value.
  `DualDecoderCombinedModule.compute_separate_losses` takes it as an optional fifth argument.
- `LocalModule._common_step` returned a 4-tuple where the training plan unpacked 5; it now
  matches the 6-tuple contract of the other modules.

### Fixed

- The masked per-cell cosine (`..._masked_cosine_similarity`) averaged over every cell, and a cell
  without a masked entry entered as a cosine of 0. Under cell masking that is every unmasked cell,
  so the logged value was scaled by the fraction of cells masked: a near-perfect reconstruction
  read 0.35 at a 30% masking rate. It now averages over the cells with a masked entry. The other
  masked metrics were not affected.
- `SCE_EntropyATT_Loss` under cell masking averaged its cosine term over every cell, with the
  unmasked ones zeroed into a constant error of 1: the loss was `f * SCE(masked) + (1 - f)` for a
  masked fraction `f`, and its gradient `f` times too small. It now averages over the masked cells,
  as `SCELoss` already did. **This changes training under that loss.**
- Under gene masking, the row-normalising losses (`SCELoss`, `SCE_EntropyATT_Loss`,
  `BalancedPearsonCorrelationLoss`) counted a cell with no masked gene as an all-zero row. Such
  cells are now left out; at typical rates and panel sizes they almost never occur.
- `optim.monitor` minimised only metrics ending in `loss`, so monitoring an error -- `val_mse`, or
  the new scaled cosine error -- made early stopping and checkpointing keep the *worst* epoch. Names
  ending in `error` or `mse` are now minimised too.

## [0.0.1]

initial release
