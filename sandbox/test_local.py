import warnings

import anndata as ad
import numpy as np
import pandas as pd

from interscale.config import load_config
from interscale.geome_dataloader import GraphAnnDataModule
from interscale.model import LocalModel
from interscale.tl import prepare_geome_dataset

warnings.filterwarnings("ignore")

rng = np.random.default_rng(0)
Xs, coords, obs = [], [], []
for s, split in zip(["s1", "s2", "s3", "s4"], ["train", "train", "val", "test"]):
    coords.append(rng.uniform(0, 120, size=(60, 2)))
    Xs.append(np.log1p(rng.poisson(2.0, size=(60, 20)).astype("float32")))
    obs.append(pd.DataFrame({"sample": [s] * 60, "split": [split] * 60}))
obs = pd.concat(obs, ignore_index=True)
obs["sample"] = obs["sample"].astype("category")
obs["split"] = obs["split"].astype("category")
adata = ad.AnnData(X=np.vstack(Xs), obs=obs)
adata.obsm["spatial"] = np.vstack(coords)
adata.obs_names = [str(i) for i in range(adata.n_obs)]

cfg = load_config("toy.yaml")
LocalModel._setup_anndata(adata, prediction_task="regression", layer_key=None,
                          sample_key_list=["sample"], split_key="split", view_registry=False)
model = LocalModel(adata, cfg=cfg)
datas, _ = prepare_geome_dataset(adata, cfg)
dm = GraphAnnDataModule(datas=datas, num_workers=0, batch_size=2,
                        mask_percentage=cfg.dataset.mask_percentage,
                        mask_strategy=cfg.dataset.mask_strategy, learning_type="node")
model.train(max_epochs=3, datamodule=dm, early_stopping=False)
print("\n>>> local component ran OK")
