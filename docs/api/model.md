# Model

InterScale is a model descigned for spatial transcpriptomics analysis. It provides, 1) **local and global embeddings** for gene level analysis and 2) **attention matrix** for cell-to-cell analysis.

## Overview

## Core components


### InterScale model

This is the main model class that can be used to define, train, and evaluate the model on an anndata.

```{eval-rst}
.. module:: interscale.model
    :no-index:
.. currentmodule:: interscale

.. autosummary::
    :nosignatures:
    :toctree: generated

    model.CombinedModel
    model.GlobalModel
    model.LocalModel
```

### InterScale module

This is the pytorch neural network module and contains InterScale logic.

## Usage example

```
import scanpy as sc
from interscale

# Load your model and training configurations
cfg = load_config(cfg_path)

# Load your data
adata = ad.read_h5ad("your_data.h5ad")

# Setup anndata
interscale.model.CombinedModel._setup_anndata(
    adata = adata,
    prediction_task = PREDICTION_TASK,
    layer_key = "norm",
    sample_key_list = ["sample"],
    prediction_obs = prediction_obs
)

# Initialize the model
model = interscale.model.CombinedModel(
    adata,
    cfg = cfg
)

pyg_data_list, _ = prepare_geome_dataset(adata, cfg)
dm = GraphAnnDataModule(datas=pyg_data_list,
                        num_workers=1,
                        batch_size=int(cfg.dataset.batch_size),
                        pct_mask_nodes=cfg.dataset.pct_mask_nodes,
                        learning_type="node")

# Train the model
model.train(max_epochs = 20,
           datamodule = dm,
           early_stopping = True,
           batch_size = int(cfg.dataset.batch_size),
           train_size = float(cfg.dataset.train_size),
           validation_size = float(cfg.dataset.val_size),
           wandb_use = False)

# Get model output
result = model.get_model_output(adata)

# Please check tutorials for more details and downstream steps
```
