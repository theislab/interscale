import warnings

warnings.filterwarnings("ignore")
import sys
from pathlib import Path

import anndata as ad
import pandas as pd

from interscale.config import load_config
from interscale.geome_dataloader import GraphAnnDataModule
from interscale.model import LocalModel
from interscale.tl import prepare_geome_dataset, set_full_reproducibility

# Usage, from the repo root:  python scripts/train_local.py configs/merfish_gcn_radius.yaml
if len(sys.argv) < 2:
    raise SystemExit("usage: python scripts/train_local.py <config.yaml>")

# 1. Config and seeds (tutorial: load_config + set_full_reproducibility)
cfg = load_config(sys.argv[1])
# sets the random number generators of Python, NumPy and PyTorch to 40
set_full_reproducibility(cfg.optim.seed)

# 2. Data (tutorial: sc.read_h5ad of the prepared file)
adata = ad.read_h5ad(cfg.dataset.h5ad_data)

# 3. Register the data and build the model (tutorial: CombinedModel; here only the local GNN)
LocalModel._setup_anndata(
    adata,
    prediction_task=cfg.dataset.prediction_task,
    prediction_obs=cfg.dataset.prediction_obs,
    layer_key=cfg.dataset.layer_key,
    sample_key_list=list(cfg.dataset.sample_key),
    split_key=cfg.dataset.split_key,
    view_registry=True,  # prints the registry tables, as in the tutorial
)
# builds the network, with its sizes taken from the registry
model = LocalModel(adata, cfg=cfg)
print(model._model_summary_string)

# 4. Graphs and data loaders (tutorial: prepare_geome_dataset + GraphAnnDataModule)
# takes that slice’s cells, connects every pair of cells closer than 30 um, packs the result into a graph
datas, _ = prepare_geome_dataset(adata, cfg)
dm = GraphAnnDataModule(
    datas=datas,
    num_workers=0,  # tutorial uses 1; 0 keeps everything in one process, which needs less memory (trying to run locally)
    batch_size=int(cfg.dataset.batch_size),
    mask_percentage=cfg.dataset.mask_percentage,  # tutorial's pct_mask_nodes was renamed to these two
    mask_strategy=cfg.dataset.mask_strategy,
    learning_type="node",
)

# 5. Train (tutorial: model.train(max_epochs=20, datamodule=dm, early_stopping=True, wandb_use=False)).
#    train() also saves the best model to `model.save` from the YAML, so no separate model.save() is needed.
model.train(max_epochs=cfg.optim.n_epochs, datamodule=dm, early_stopping=cfg.optim.early_stopping, wandb_use=False)

# 6. Save the metrics of every epoch (tutorial plots them instead). The filename has dataset name, GNN and seed,
#    so runs never overwrite each other.
history = pd.DataFrame(model.history_.history)
out = Path("results") / f"history_{cfg.dataset.name}_{cfg.model.local_component.name}_seed{cfg.optim.seed}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
history.to_csv(out, index=False)
print(f"\n>>> trained OK; metrics per epoch saved to {out}")
