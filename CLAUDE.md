# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

InterScale is a computational model for analysis of intercellular interactions in spatial transcriptomics across different length-scales. It combines a local GNN component (GCN/GIN) and a global Transformer component to generate per-cell embeddings and attention matrices for tissue, cell, and gene-level communication analysis.

## Commands

### Installation

```bash
# Recommended (uv)
uv venv .interscale --python 3.13
source .interscale/bin/activate
uv pip install torch torchvision torchaudio
uv pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.10.0+cpu.html
uv pip install torch-geometric pytorch-lightning wandb yacs scvi-tools
uv pip install geome
uv pip install -e .
```

### Linting

```bash
ruff check src/        # check for issues
ruff check --fix src/  # auto-fix
ruff format src/       # format code
pre-commit run --all-files  # run all pre-commit hooks
```

### Testing

```bash
pytest                                 # run all tests
pytest tests/test_geome_dataloader.py  # run a single test file
pytest tests/ -k "test_name"           # run a specific test by name
```

### Training

```bash
python src/InterScale/main.py --cfg "path/to/config.yaml" --model_type "CombinedModel"
# model_type options: LocalModel, GlobalModel, CombinedModel
```

## Architecture

### Two-scale design

The core idea is a two-component architecture:

1. **Local component** (`module/local_modules/`): A GNN (currently GCN or GIN) that processes per-cell gene expression and its immediate spatial neighbors via `edge_index`. Outputs local embeddings `[N, n_embed]`.

2. **Global component** (`module/global_modules/`): A Transformer encoder (`TransformerNodeEncoderHook`) that receives the local embeddings padded into sequences of length `max_seq_len` (one sequence per tissue sample/FOV/sliding window). A CLS token is appended. Outputs global embeddings and a self-attention matrix capturing long-range cell-cell interactions.

3. **CombinedModel** is the primary model combining both. `LocalModel` and `GlobalModel` run each component independently.

### Data flow

```
AnnData (h5ad) → squidpy spatial graph → geome PyG Data objects
→ GraphAnnDataModule (Lightning DataModule)
→ [node masking ~20% of cells]
→ LocalModule.forward(x, edge_index) → local_embedding [N, n_embed]
→ GlobalModule.common_step_local_to_global() → padded_emb [max_seq_len, B, n_embed]
→ TransformerEncoder → global_embedding + attn_matrix
→ Decoder → y_pred (regression: GEX values, or classification: cell type)
```

### Key classes

| Class | File | Role |
|---|---|---|
| `CombinedModel` | `model/CombinedModel.py` | Top-level model; inherits `BaseModelClass` + `NodeMaskingTrainingPlan` |
| `BaseModelClass` | `model/base/_base_model.py` | AnnData management (scvi-tools pattern), save/load, `_setup_anndata` |
| `NodeMaskingTrainingPlan` | `train/_training.py` | `model.train()` entry point; wraps PyTorch Lightning Trainer |
| `TrainingPlan` | `train/_trainingplans.py` | Lightning module defining optimizer, train/val steps, losses |
| `CombinedModuleClass` | `module/combined_module/combined_module.py` | Wires local + global modules, implements `forward` and `_common_step` |
| `LocalModuleClass` | `module/base/_base_local_module.py` | Abstract base for GCN, GIN, scVI, Precomputed |
| `GlobalModuleClass` | `module/base/_base_global_module.py` | Abstract base for Transformer encoder |
| `TransformerNodeEncoderHook` | `module/global_modules/transformer_encoder.py` | Transformer with CLS token and self-attention relevance hooks |
| `GraphAnnDataModule` | `geome_dataloader.py` | Lightning DataModule; wraps geome PyG Data lists |

### Configuration system

Config uses [YACS](https://github.com/rbgirshick/yacs). Defaults are defined per-module in `config/`:
- `dataset_config.py` — data paths, sample keys, masking fraction, train/val/test splits
- `model_config.py` — `n_embed`, local/global component names, decoder type
- `local_component_config.py` — GCN/GIN hyperparams (n_layers, hidden_dim, dropout)
- `global_component_config.py` — Transformer hyperparams (max_seq_len, n_heads, num_layers, dim_feedforward)
- `optim_config.py` — lr, weight_decay, loss, scheduler, early stopping, accelerator
- `wandb_config.py` — wandb project name, logging toggle

Override defaults by providing a `.yaml` file (see `src/config_files/InterScale_example.yaml`). Only keys that differ from defaults need to be specified.

### AnnData conventions

The model follows scvi-tools AnnDataManager patterns. Before instantiating a model, call `Model._setup_anndata(adata, ...)`. The AnnData must have:
- `adata.X` or a layer: normalized gene expression (log-norm recommended, values 0–3)
- `adata.obsp["spatial_connectivities"]`: squidpy spatial neighbor graph
- `adata.obs["split"]`: train/val (and optionally test) split labels
- `adata.obs[sample_key]`: grouping key for splitting into PyG Data objects (FOV, sliding window, etc.)
- `adata.obsm["spatial"]`: spatial coordinates (required for `get_model_output`)

After training, results are stored back in adata:
- `adata.obsm["{prefix}_local_emb"]`, `adata.obsm["{prefix}_global_emb"]`
- `adata.obsm["{prefix}_attn_matrix"]`, `adata.obs["{prefix}_cls_horizontal/vertical"]`
- `adata.layers["{prefix}_y_pred_global"]` (regression) or `adata.obsm["{prefix}_y_pred_global"]` (classification)

### Module layout

```
src/InterScale/
├── config/          # YACS config defaults
├── evaluation/      # clustering, gene ranking, graph classification metrics
├── model/           # LocalModel, GlobalModel, CombinedModel (high-level API)
│   └── base/        # BaseModelClass (save/load, AnnData setup)
├── module/          # Neural network modules
│   ├── base/        # Abstract base classes
│   ├── combined_module/  # CombinedModuleClass, DualDecoderCombinedModuleClass
│   ├── global_modules/   # TransformerNodeEncoderHook, CustomTransformerEncoderLayer
│   └── local_modules/    # GCN, GIN, SCVI, Precomputed
├── nn/              # Low-level building blocks (_base_components.py)
├── pp/              # Preprocessing (spatial data, segmentation noise)
├── tl/              # Tools: masking, padding, patient split, scheduler, utils
└── train/           # TrainingPlan (Lightning), losses, training utilities
```
