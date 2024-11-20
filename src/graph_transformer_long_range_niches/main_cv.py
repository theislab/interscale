from graph_transformer_long_range_niches.model import LitGNNTransformer, LitPCATransformer, BaselineFCNN
from graph_transformer_long_range_niches.pp.geome_utils import prepare_geome_dataset, split_adata
from graph_transformer_long_range_niches.modules.gcn import LitGCN
from graph_transformer_long_range_niches.tl.wandb import log_data
from graph_transformer_long_range_niches.config import load_config

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

import argparse
import wandb
from torch_geometric.data.lightning import LightningDataset
import math
import scanpy as sc
import os
import pickle
from sklearn.model_selection import KFold

import json

def main(cfg_path):

    cfg = load_config(cfg_path)

    ####### PREPROCESSING #######
    # Load adata
    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    adata.obs_names_make_unique()

    # Initialize KFold
    n_splits = cfg.dataset.k_folds  # Set the number of folds in your config file
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.optim.seed)

    # Split data into train, val (and test)
    train_size, val_size, test_size = float(cfg.dataset.train_size), float(cfg.dataset.val_size), float(cfg.dataset.test_size)
    adata = split_adata(adata, split_obs=cfg.dataset.obs_split, val_size=val_size, test_size=test_size, seed = cfg.optim.seed, k_splits=n_splits)

    for fold in range(n_splits):

        # Create PyG data
        print('Load PyG data...')
        pyg_data_list, _ = prepare_geome_dataset(adata, cfg, split_key=f'split_{fold+1}')

        # Cross-Validation Loop
        fold_results = []

        print(f"Starting fold {fold + 1}/{n_splits}...")
        # Split data for this fold
        train_ds, val_ds = pyg_data_list[0], pyg_data_list[1]
        # set number of classes and number of features
        cfg.dataset.merge_from_list(['num_features', len(train_ds[0].x[1])])
        if 'classification' in cfg.dataset.prediction_task:
            cfg.dataset.merge_from_list(['num_classes', len(train_ds[0].y[1])])

        if test_size > 0.0:
            test_ds = pyg_data_list[2]
            dm = LightningDataset(train_dataset = train_ds,
                                val_dataset = val_ds,
                                test_dataset = test_ds, 
                                batch_size=int(cfg.dataset.batch_size), 
                                shuffle=True)
            print(f'train ds: {len(train_ds)}, val ds: {len(val_ds)}, test ds: {len(test_ds)}')
            datasets = [train_ds, val_ds, test_ds]
            names = ["training", "validation", "test"]
        else:
            dm = LightningDataset(train_dataset = train_ds, 
                                val_dataset = val_ds, 
                                batch_size=int(cfg.dataset.batch_size), 
                                shuffle=True)
            print(f'train ds: {len(train_ds)}, val ds: {len(val_ds)}')
            datasets = [train_ds, val_ds]
            names = ["training", "validation"]

        ####### TRAINING #######
        # Model Initialization
        try:
            if cfg.model.model_type == 'gnn-transformer':
                print('Load GNNTransfomer...')
                model = LitGNNTransformer(cfg)
            elif cfg.model.model_type == 'gnn':
                print('Load GNN...')
                model = LitGCN(cfg)
            elif cfg.model.model_type == 'fcnn':
                print('Load FCNN...')
                model = BaselineFCNN(cfg)
            elif cfg.model.model_type == 'pca-transformer':
                print('Load PCA Transformer...')
                model = LitPCATransformer(cfg)
        except ValueError:
            print("No valid model defined in .yaml file.")

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        if 'classification' in cfg.dataset.prediction_task:
            early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=10, verbose=False, mode="max")
        if 'regression' in cfg.dataset.prediction_task:
            early_stop_callback = EarlyStopping(monitor="val_r2", min_delta=0.00, patience=10, verbose=False, mode="min")

        steps_per_epoch = math.ceil(len(train_ds) / cfg.dataset.batch_size)

        data_name = f"{cfg.dataset.name}_{cfg.dataset.prediction_obs}_{cfg.dataset.library_key[-1]}_{len(cfg.dataset.library_key)}_{cfg.optim.seed}"
        run_name = f"{data_name}_{cfg.model.model_type}"

        print('Training...')
        trainer = pl.Trainer(min_epochs=1, 
                        max_epochs=int(cfg.model.n_epochs),
                        enable_progress_bar=False,
                        callbacks=[lr_monitor, early_stop_callback],
                        log_every_n_steps=steps_per_epoch,
                        )

        trainer.fit(model, dm)
        val_results = trainer.validate(model, dm)
        fold_results.append(val_results)

        del pyg_data_list, model, trainer, dm, datasets, train_ds, val_ds

        # Aggregate results across all folds
        print("Cross-validation results:", fold_results)

    output_path = cfg.model.output_path  # Assuming this path is defined in the config
    print(f"Saving model locally to {output_path}...")

    data_name = f"{cfg.dataset.name}_{cfg.dataset.prediction_obs}_{cfg.dataset.library_key[-1]}_{len(cfg.dataset.library_key)}_{cfg.optim.seed}"
    run_name = f"{data_name}_{cfg.model.model_type}"

    results_file = os.path.join(output_path, f"{run_name}_fold_results.json")

    with open(results_file, 'w') as f:
        json.dump(fold_results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GTLongRange')

    parser.add_argument('--cfg', dest='cfg', type=str, required=True, help='The configuration file path.')
    args = parser.parse_args()

    main(args.cfg)

