from graph_transformer_long_range_niches.model import LitGNNTransformer, LitPCATransformer, BaselineFCNN
from graph_transformer_long_range_niches.pp.geome_utils import prepare_geome_dataset, split_adata
from graph_transformer_long_range_niches.modules import LitGCN
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
import numpy as np

from sklearn.utils.class_weight import compute_class_weight

from graph_transformer_long_range_niches.pp import prepare_geome_dataset
from torch_geometric.loader import DataLoader

def main(cfg_path):

    cfg = load_config(cfg_path)

    ####### PREPROCESSING #######
    # Load adata
    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    adata.obs_names_make_unique()

    #adata = sliding_windows(adata, 800, library_key = 'library_key',  overlap=0)

    # Split data into train, val (and test)
    train_size, val_size, test_size = float(cfg.dataset.train_size), float(cfg.dataset.val_size), float(cfg.dataset.test_size)
    if 'graph' in cfg.dataset.prediction_task:
        split_adata(adata, split_obs=cfg.dataset.obs_split, val_size=val_size, test_size=test_size, seed = cfg.optim.seed, stratify_groups = cfg.dataset.prediction_obs)
    elif cfg.dataset.stratify_group is not None:
        print('Stratifying by group: ', cfg.dataset.stratify_group)
        split_adata(adata, split_obs=cfg.dataset.obs_split, val_size=val_size, test_size=test_size, seed = cfg.optim.seed, stratify_groups = cfg.dataset.stratify_group)
    else:
        split_adata(adata, split_obs=cfg.dataset.obs_split, val_size=val_size, test_size=test_size, seed = cfg.optim.seed)

    if cfg.optim.loss == 'WeightedCE':
        class_weigths = compute_class_weight("balanced", classes = np.unique(adata.obs[cfg.dataset.prediction_obs]), y=adata.obs[cfg.dataset.prediction_obs])
        print("WeightedCE with class weights: ", class_weigths)
    else: 
        class_weigths = None

    # Create PyG data
    print('Load PyG data...')
    pyg_data_list, adata_list = prepare_geome_dataset(adata, cfg)
    train_ds, val_ds = pyg_data_list[0], pyg_data_list[1]
    # set number of classes and number of features
    cfg.dataset.merge_from_list(['num_features', len(train_ds[0].x[1])])
    if 'classification' in cfg.dataset.prediction_task:
      cfg.dataset.merge_from_list(['num_classes', len(train_ds[0].y[1])])
    # Initialize Dataset for training
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
            model = LitGCN(cfg, class_weigths)
        elif cfg.model.model_type == 'fcnn':
            print('Load FCNN...')
            model = BaselineFCNN(cfg)
        elif cfg.model.model_type == 'pca-transformer':
            print('Load PCA Transformer...')
            model = LitPCATransformer(cfg, class_weigths)
    except ValueError:
        print("No valid model defined in .yaml file.")

    steps_per_epoch = math.ceil(len(train_ds) / cfg.dataset.batch_size)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    if 'classification' in cfg.dataset.prediction_task:
        early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.05, patience=10*steps_per_epoch, verbose=False, mode="max")
    elif 'regression' in cfg.dataset.prediction_task:
        early_stop_callback = EarlyStopping(monitor="val_r2", min_delta=0.05, patience=10*steps_per_epoch, verbose=False, mode="min")
    else:
        raise Exception("Training must be classification or regression based.")
    
    data_name = f"{cfg.dataset.name}_{cfg.dataset.prediction_obs}_{cfg.dataset.library_key[-1]}_{len(cfg.dataset.library_key)}_{cfg.optim.seed}"
    run_name = f"{data_name}_{cfg.model.model_type}"

    if cfg.wandb.use:
        print('Wandb initialize...')
        run = wandb.init(project=cfg.wandb.project_name, config=cfg, name=run_name, job_type = 'model_training')
        #log_data(datasets + adata_list, names + ['adata'], cfg, run)
        #cfg._data = wandb.config # make sure that what is logged is same as waht is run
        wandb_logger = WandbLogger(name = run_name, log_model=True) #save at the end of the training
        if 'classification' in cfg.dataset.prediction_task:
            checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max", filename=run_name) # save model if validation accuracy increases
        elif 'regression' in cfg.dataset.prediction_task:
            if cfg.optim.loss == 'MSELoss':
                checkpoint_callback = ModelCheckpoint(monitor="val_mse", mode="min", filename=run_name) 
            elif cfg.optim.loss == 'GaussianNLL' or cfg.optim.loss == 'SmoothL1':
                checkpoint_callback = ModelCheckpoint(monitor="val_r2", mode="max", filename=run_name) 
            else:
                raise Exception("Regression must be run with MSELoss, GaussianNLL or SmoothL1 loss.")
        else:
            raise Exception("Training must be classification or regression based.")
        
        print('Training Wandb...')
        trainer = pl.Trainer(min_epochs=1, 
                         max_epochs=int(cfg.model.n_epochs), 
                         logger=wandb_logger, 
                         enable_progress_bar=False, 
                         callbacks=[lr_monitor, checkpoint_callback, early_stop_callback],
                         log_every_n_steps=steps_per_epoch,
                         # Sanity checks: Debugging model
                         #overfit_batches=1,
                         )
    else:
        print('Training...')
        trainer = pl.Trainer(min_epochs=1, 
                         max_epochs=int(cfg.model.n_epochs),
                         enable_progress_bar=False,
                         callbacks=[lr_monitor, early_stop_callback],
                         log_every_n_steps=steps_per_epoch,
                         # Sanity checks: Debugging model
                         #overfit_batches=1,
                         )

    trainer.fit(model, dm)
    trainer.validate(model, dm)
    if test_size > 0.0:
        trainer.test(model, dm)

    ##### SAVING #####
    if cfg.model.save == "local":
        output_path = cfg.model.output_path  # Assuming this path is defined in the config
        print(f"Saving model locally to {output_path}...")
        trainer.save_checkpoint(os.path.join(output_path, f"{run_name}.ckpt"))

        adata.obs.to_csv(os.path.join(output_path,f'{data_name}_obs.csv'), index=True)

        with open(os.path.join(output_path, f"{data_name}.pkl"), 'wb') as file:
            pickle.dump(datasets, file)

    if cfg.model.save == "wandb" and cfg.wandb.use:
        ## log model artifact
        model_checkpoint_path = checkpoint_callback.best_model_path

        if model_checkpoint_path:
            # Create an artifact
            artifact = wandb.Artiftact(name=f"{run_name}_model_{cfg.optim.seed}", type="model")
            artifact.add_file(model_checkpoint_path, name=f"{run_name}.ckpt")

            # Log the artifact
            run.log_artifact(artifact)

        run.finish()
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GTLongRange')

    parser.add_argument('--cfg', dest='cfg', type=str, required=True, help='The configuration file path.')
    args = parser.parse_args()

    main(args.cfg)