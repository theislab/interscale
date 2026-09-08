# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

InterScale is a Python package for multi-scale cell interaction analysis in spatial transcriptomics data. It combines a local, graph-based component (cell-neighborhood scale) with a global, transformer-based component (whole-sample scale) into a single trainable model. It is built on top of `geome`/PyTorch Geometric, AnnData, scanpy/squidpy, `scvi-tools` (for AnnData registration/data management), and `lightning`/`pytorch_lightning` for training.

For the scientific motivation, model design rationale, and terminology behind this architecture (from the InterScale preprint — Drummer, Jiménez et al., bioRxiv 2026), see [`.claude/background.md`](.claude/background.md). Read it before working on model/module architecture, loss functions, or the interpretability/evaluation pipeline — it explains *why* the local/global split and dual-decoder design exist, not just what the code does.

## Common commands

This project uses `hatch` as the primary project manager (also works with `uv` or `pip`; see `docs/installation.md`).

```bash
# Install dev dependencies (uv)
uv sync --all-extras

# Install dev dependencies (pip)
pip install -e ".[dev,test,doc]"

# Run the full test suite
hatch test            # or: uv run pytest
hatch test --all      # across the full Python version matrix

# Run a single test file / test
uv run pytest tests/test_package.py
uv run pytest tests/test_package.py::test_import

# Lint / format (ruff + biome are also run via pre-commit)
uv run ruff check .
uv run ruff format .
pre-commit run --all-files

# Build docs
hatch run docs:build
hatch run docs:open
```

Training entrypoints are run as scripts, not via a CLI command:

```bash
python src/interscale/main.py --cfg config_files/legnini_example.yaml --model_type CombinedModel
python src/interscale/main_sweep.py --cfg <cfg.yaml> --sweep_cfg <sweep.yaml> --model_type CombinedModel --sweep_goal hyperparmeter
```

`--model_type` is one of `LocalModel`, `GlobalModel`, `CombinedModel`. `main_sweep.py` additionally requires `wandb` and a sweep config with `sweep_goal` in `{robustness, segmentation, hyperparmeter, loss}`.

Ruff config lives in `pyproject.toml` (`[tool.ruff]`): line length 120, numpy docstring convention, many rules relaxed for legacy code (see `lint.ignore`). Tests are exempt from docstring rules (`D`).

## Architecture

### Config-driven design (yacs)

Everything is driven by a single `yacs` `CfgNode` (`cfg`), built via `interscale.config.load_config(cfg_path)`:
- `get_cfg_defaults()` assembles defaults from `config/{wandb,model,optim,dataset}_config.py`.
- `load_config` additionally loads component-specific defaults based on `cfg.model.local_component.name` / `cfg.model.global_component.name` (from `local_component_config.py` / `global_component_config.py`) *before* merging the user's YAML file, then freezes the config.
- User-facing YAML files (see `config_files/legnini_example.yaml`) only need to override values relevant to their run — everything else falls back to defaults.
- The `cfg` object is threaded through nearly every layer (model, module, dataloader, geome dataset prep) rather than passed as individual kwargs.

### Model hierarchy (`interscale.model`)

`BaseModel` (`model/base/_base_model.py`) mimics the `scvi-tools` model API pattern:
- Uses an `scvi.data.AnnDataManager` (registered via `_setup_anndata`/`register_fields`) to validate and access fields on the `AnnData` object (expression layer, sample/library keys, prediction obs, split key). This is why models require `ModelClass._setup_anndata(...)` to be called before instantiation — it registers the manager keyed by an AnnData UUID stored in `adata.uns`.
- `prediction_task` is `"classification"` or `"regression"`; `prediction_level` is `"node"` or `"graph"` — these gate a lot of branching throughout the codebase (loss selection, decoder output shape, evaluation output storage).
- `save`/`load` persist only the module `state_dict`, with filenames derived from `tl.utils.get_model_filename_prefix(cfg, local_component, global_component)`. `load` includes legacy state-dict key remapping (`global_module.module.` → `global_module.`, stripped `module.` prefixes, and `tl.utils.detect_and_remap_state_dict_keys`) to stay compatible with older/wandb-saved checkpoints.

Three concrete models subclass `BaseModel`, differing in which components they instantiate and which module class they wrap:
- `LocalModel` — local (graph) component only.
- `GlobalModel` — global (transformer) component only.
- `CombinedModel` — both; wraps either `CombinedModule` or `DualDecoderCombinedModule` (when `cfg.model.decoder.dual_decoder` is `True`, both local and global components get their own decoder instead of sharing one).

`BaseModel._register_local_component` / `_register_global_component` look up `cfg.model.local_component.name` / `cfg.model.global_component.name` and instantiate the matching module class (currently `"GCN"` and `"self-attn-transformer"` respectively) — adding a new local/global component means adding a branch here plus a corresponding config file.

### Module hierarchy (`interscale.module`)

Mirrors the model hierarchy at the `pytorch_lightning`/`LightningModule` level:
- `BaseModule` (`module/base/_base_module.py`) owns the decoder (`interscale.nn`: `LinearDecoder`, `LinearLSEDecoder`, `NonLinearDecoder`, or `None` when a wrapping combined module owns decoding), and shared masking logic (`tl.masking.apply_mask`, via `_common_step_masking`) used for self-supervised/robustness training. `_common_step_masking` returns `(batch_masked, mask_idx, entry_mask)`; every `_common_step` returns `(local_emb, global_emb, y_pred, y_true, attn, entry_mask)`, and `entry_mask` is what restricts the loss and metrics under gene masking.
- `module/local_modules/` — local (graph) encoders: `GCN`, `GIN`, `Precomputed` (precomputed embeddings), `SCVI` (SCVI-based encoder).
- `module/global_modules/` — transformer-based global encoder (`TransformerNodeEncoderHook`) plus supporting transformer encoder/layer/utils, encoding whole-sample (long-range) context across cells.
- `module/combined_module/` — `CombinedModule` and `DualDecoderCombinedModule` compose a local module + global module sequentially: local embeddings feed into the global (transformer) component, and predictions/attention/CLS tokens are extracted from there (see `CombinedModel.get_model_output` in `model/combined_model.py` for the full inference/evaluation flow, including how attention matrices and horizontal/vertical CLS tokens are extracted and padded to `max_seq_len`).

### Data pipeline: AnnData → PyG graphs

`interscale.tl.geome_utils.prepare_geome_dataset` / `prepare_a2d_dataset` bridge AnnData and PyTorch Geometric using `geome`:
- Iterates per-sample/library (`cfg.dataset.sample_key`), builds a spatial neighbor graph per sample (`squidpy`-style `spatial_neigbors_kwargs`, converted to an edge index via `geome.transforms.AddEdgeIndex`), and yields one PyG `Data` object per sample.
- Splits data by `cfg.dataset.split_key` (an `adata.obs` column that must contain `train`/`val`, optionally `test`) — this must exist in the AnnData before calling `prepare_geome_dataset`.
- Handles both classification (one-hot encodes `prediction_obs`) and regression prediction tasks, and optionally attaches precomputed embeddings (`cfg.model.global_component.parameters.type_gex_embedding == "Precomputed"`) from `adata.obsm`.
- `interscale.geome_dataloader.GraphAnnDataModule` wraps the resulting `list[Data]` splits into a `LightningDataModule`. For node-level learning it corrupts the reconstruction input for each dataloader construction (redrawn every epoch by `NodeMaskResampleCallback`) — this is the masking scheme used for self-supervised node reconstruction/robustness experiments referenced in `config_files/legnini_example.yaml`'s `pct_mask_nodes` and `dataset.segmentation_robustness`.

### Masking (`tl/masking.py`)

Two config axes, both under `cfg.dataset`. Read the module docstring of `tl/masking.py` before changing any of them.

| key | values | meaning |
| --- | --- | --- |
| `mask_strategy` | `node` (default) / `gene` | Granularity. `node` blanks whole cells (`data.mask` `[N]`) and scores all G genes of them. `gene` blanks individual `(cell, gene)` entries (`data.gene_mask` `[N, G]`, every cell a target) and scores **only those entries**. |
| `mask_percentage` | float | The Bernoulli rate, per cell under `node` and per entry under `gene` — one key, because the strategy already says what a unit is. Not comparable across strategies at equal values. `_validate_masking` rejects a regression config with a rate outside (0, 1]. |

Masked positions are always filled with `tl.masking.MASK_VALUE`, which is **-1** and is not
configurable. Two things are easy to get wrong here:

- **The strategy is passed explicitly, never inferred.** `apply_mask` and `_process_batch_for_metrics` both take/consult `mask_strategy` rather than checking whether a `gene_mask` attribute exists, so a stale mask on a reused `Data` object cannot turn a cell-masking run into a gene-masking one. That is why the dataloader has no clear-the-attribute helper.
- **The loss must be restricted to the masked entries under `gene`.** That is what `entry_mask` threads through `_common_step` → `TrainingPlan._regression_metrics` → `tl.masking.masked_loss`, and metrics go through `train._trainingplans.masked_regression_metrics` (same metric *names*, computed over masked entries only, verified against torchmetrics in `tests/test_gene_masking.py`). Without it the unmasked entries — which were handed to the model as input — make the objective the identity map. **Metric names are shared but the entry sets differ, so a `val_r2` from a `node` run and one from a `gene` run are not comparable as absolute numbers.**
- **`MASK_VALUE` must stay outside the layer's range; do not set it back to 0.** ~62% of legnini23's `log1p_norm` entries are already exactly 0 (and 648 cells are all-zero), so a zero fill makes a masked position indistinguishable from a real measurement — under `gene` the corruption becomes invisible and the model cannot tell which entries it is being asked for. Cell masking was less exposed because an all-zero *row* is still mostly a pattern. Measured at gene rate 0.25 over 3 seeds: fill 0 gave `val_concordance_corr` 0.0696 ±0.0113, fill -1 gave 0.0799 ±0.0046 — better on every seed and with 2.5× less run-to-run spread, below the cell-masking baseline's own CV. A learnable per-gene `[MASK]` token (GraphMAE's design) was tried and tied with the fixed -1 to three decimals — it drifts only ~0.02–0.035 from any init — so it was removed rather than kept as an option.

The registered ablation pairs are `<dataset>/node_reg` (cell) vs `<dataset>/node_reg_genemask` (gene) — note each carries its own `dataset.name` so the two arms do not share a checkpoint prefix. `config_files/sweeps/mask_granularity_legnini.yaml` runs the granularity/rate grid.

### Preprocessing / robustness utilities (`interscale.pp`)

`pp.segmentation_noise.apply_segmentation_noise` simulates cell-segmentation errors (used when `cfg.dataset.segmentation_robustness` is set in `main.py`/`main_sweep.py`) for robustness sweeps.

### Datasets (`interscale.datasets`)

- `datasets/_legnini.py` provides `legnini()`, a loader for the Legnini et al. 2023 molecular cartography dataset used in the manuscript/tutorials — downloads from Zenodo and caches to `~/.cache/interscale/legnini_2023.h5ad`.


### Evaluation (`interscale.evaluation`)

Post-hoc analysis utilities operating on the `adata` produced by `save_evaluation_results`/`get_model_output`: gene loadings, gene-rank analysis, gene-set covariance, latent-space analysis, graph classification metrics, and network/attention stream visualization (`net_streams.py`).

## Repository conventions

- Numpy-style docstrings (see `docs/contributing.md`); many docstring lint rules are intentionally disabled in `pyproject.toml` for legacy modules, but new public APIs should still be documented in numpy style since `sphinx-autodoc-typehints`/napoleon render them for the docs site.
- Code style is enforced by `pre-commit` (ruff-check/format, biome-format, pyproject-fmt, whitespace/merge-conflict hooks) — run `pre-commit install` once locally.
- Tests live under `tests/`, use `pytest` with `--import-mode=importlib`; shared fixtures (e.g. a small synthetic `AnnData`) are defined in `conftest.py` at the repo root.
- Coverage is measured over the `interscale` package only (`[tool.coverage] run.source = ["interscale"]`), excluding `test_*.py` files.
