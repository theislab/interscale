from graph_transformer_long_range_niches.model import LitGNNTransformer, LitPCATransformerMasked, BaselineFCNN, LitGNNTransformerMasked
from graph_transformer_long_range_niches.pp import prepare_geome_dataset, split_adata
from graph_transformer_long_range_niches.modules import LitGCN, LitGCNMasked
from graph_transformer_long_range_niches.tl import MaskedNodeLightningDataset
from graph_transformer_long_range_niches.tl.geome_dataloader import GraphAnnDataModule
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

import yaml

from sklearn.utils.class_weight import compute_class_weight

def main_sweep(cfg_path, sweep_goal):
    
    cfg = load_config(cfg_path)

    data_name = f"{cfg.dataset.name}_{cfg.dataset.prediction_obs}_{cfg.dataset.library_key[-1]}_{len(cfg.dataset.library_key)}_{cfg.optim.seed}"
    run_name = f"{data_name}_{cfg.model.model_type}"    

    if cfg.wandb.use:
        print('Wandb initialize...')
        sweep_run = wandb.init(project=cfg.wandb.project_name, 
                         config=cfg, 
                         name=run_name, 
                         job_type = 'model_training')
        sweep_config = wandb.config
        
    # Update configuration with sweep parameters
    if sweep_config is not None:
        cfg.set_new_allowed(True)
        cfg.defrost()
        print('sweep config: ', sweep_config)
        print('sweep run: ', sweep_run.config)
        if sweep_goal == 'hyperparmeter':
            print('hyperparameter sweep')
            cfg.model.lr = sweep_config.parameters.model.lr
            cfg.model.n_epochs = sweep_config['model.n_epochs']
            #cfg.dataset.batch_size = sweep_run.config.dataset.batch_size
            cfg.dataset.warm_up = sweep_config['optim.warm_up']
            cfg.dataset.wd = sweep_config['optim.wd']
            if cfg.model.model_type == 'gnn-transformer' or cfg.model.model_type == 'gnn':
                print('gnn configs')
                cfg.gnn.num_layers = sweep_config['gnn.num_layers']
                cfg.gnn.hidden_dim = sweep_config['gnn.hidden_dim']
                cfg.gnn.embed_dim = sweep_config['gnn.embed_dim']
                cfg.gnn.dropout = sweep_config['gnn.dropout']
            if 'transformer' in cfg.model.model_type:
                print('transformer configs')
                cfg.transformer.d_model = sweep_config['gnn.embed_dim'] # input transformer dimension equal to gnn embed dim
                cfg.transformer.dim_feedforward = sweep_config['transformer.dim_feedforward']
                cfg.transformer.num_layers = sweep_config['transformer.num_layers']
                cfg.transformer.n_heads = sweep_config['transformer.n_heads']
                cfg.transformer.dropout = sweep_config['transformer.dropout']
                #cfg.transformer.max_seq_len = sweep_run.config.transformer.max_seq_len
        elif sweep_goal == 'parameter':
            print('parameter space sweep')
            if 'gnn' in cfg.model.model_type:
                print('gnn configs')
                cfg.gnn.num_layers = sweep_config['gnn.num_layers']
                cfg.gnn.hidden_dim = sweep_config['gnn.hidden_dim']
                cfg.gnn.embed_dim = sweep_config['gnn.embed_dim']
                assert cfg.gnn.num_layers == sweep_config['gnn.num_layers'] and cfg.gnn.hidden_dim == sweep_config['gnn.hidden_dim'] and cfg.gnn.embed_dim == sweep_config['gnn.embed_dim']
            if 'transformer' in cfg.model.model_type:
                print('transformer configs')
                cfg.transformer.d_model = sweep_config['gnn.embed_dim'] # input transformer dimension equal to gnn embed dim
                cfg.transformer.dim_feedforward = sweep_config['transformer.dim_feedforward']
                cfg.transformer.num_layers = sweep_config['transformer.num_layers']
                cfg.transformer.n_heads = sweep_config['transformer.n_heads']
                assert cfg.transformer.d_model == sweep_config['gnn.embed_dim'] and cfg.transformer.num_layers == sweep_config['transformer.num_layers']
        elif sweep_goal == 'robustness':
            print('robustness sweep')
            cfg.dataset.spatial_neigbors_kwargs.radius = sweep_config['dataset.spatial_neigbors_kwargs.radius']
            cfg.dataset.split_key = sweep_config['dataset.split_key']
            cfg.dataset.pct_mask_nodes =  sweep_config['dataset.pct_mask_nodes']
            cfg.model.decoder.type = sweep_config['model.decoder.type']
        elif sweep_goal == 'experiment':    
            print('experiment sweep')
            cfg.optim.seed = sweep_config['optim.seed']        
            cfg.dataset.spatial_neigbors_kwargs.radius = sweep_config['dataset.spatial_neigbors_kwargs.radius']
            cfg.transformer.pct_mask_nodes = sweep_config['transformer.pct_mask_nodes']
            cfg.model.decoder.type = sweep_config['model.decoder.type']
        cfg.freeze()

        
    ####### PREPROCESSING #######
    # Load adata
    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    adata.obs_names_make_unique()

    #adata = sliding_windows(adata, 800, library_key = 'library_key',  overlap=0)

    # Split data into train, val (and test)
    train_size, val_size, test_size = float(cfg.dataset.train_size), float(cfg.dataset.val_size), float(cfg.dataset.test_size)
    if cfg.dataset.split_key in adata.obs.columns:
        print('Split already exists in adata.obs')
    else:
        if 'graph' in cfg.dataset.prediction_task:
            split_adata(adata, split_obs=cfg.dataset.obs_split, val_size=val_size, test_size=test_size, seed = cfg.optim.seed, split_key = cfg.dataset.split_key, stratify_groups = cfg.dataset.prediction_obs)
        elif cfg.dataset.stratify_group is not None:
            print('Stratifying by group: ', cfg.dataset.stratify_group)
            split_adata(adata, split_obs=cfg.dataset.obs_split, val_size=val_size, test_size=test_size, seed = cfg.optim.seed, split_key = cfg.dataset.split_key, stratify_groups = cfg.dataset.stratify_group)
        else:
            split_adata(adata, split_obs=cfg.dataset.obs_split, val_size=val_size, test_size=test_size, seed = cfg.optim.seed, split_key = cfg.dataset.split_key)

    if cfg.optim.loss == 'WeightedCE':
        class_weigths = compute_class_weight("balanced", classes = np.unique(adata.obs[cfg.dataset.prediction_obs]), y=adata.obs[cfg.dataset.prediction_obs])
        print("WeightedCE with class weights: ", class_weigths)
    else: 
        class_weigths = None

    # Create PyG data
    print('Load PyG data...')
    pyg_data_list, _ = prepare_geome_dataset(adata, cfg)
    train_ds, val_ds = pyg_data_list[0], pyg_data_list[1]
    # set number of classes and number of features
    cfg.dataset.merge_from_list(['num_features', len(train_ds[0].x[1])])
    if 'classification' in cfg.dataset.prediction_task:
      cfg.dataset.merge_from_list(['num_classes', len(train_ds[0].y[1])])
    # Initialize Dataset for training
    if test_size > 0.0:
        test_ds = pyg_data_list[2]
        if cfg.dataset.pct_mask_nodes > 0:
            dm = GraphAnnDataModule(datas=pyg_data_list, 
                           num_workers=1, 
                           batch_size=int(cfg.dataset.batch_size), 
                           pct_mask_nodes=cfg.dataset.pct_mask_nodes,
                           learning_type="node")
        else:   
            dm = MaskedNodeLightningDataset(train_dataset = train_ds,
                                val_dataset = val_ds,
                                test_dataset = test_ds, 
                                batch_size=int(cfg.dataset.batch_size), 
                                pct_mask_nodes=cfg.dataset.pct_mask_nodes,
                                shuffle=True)
        print(f'train ds: {len(train_ds)}, val ds: {len(val_ds)}, test ds: {len(test_ds)}')
        datasets = [train_ds, val_ds, test_ds]
        names = ["training", "validation", "test"]
    else:
        if cfg.dataset.pct_mask_nodes > 0:
            dm = GraphAnnDataModule(datas=pyg_data_list, 
                        num_workers=1, 
                        batch_size=int(cfg.dataset.batch_size), 
                        pct_mask_nodes=cfg.dataset.pct_mask_nodes,
                        learning_type="node")
        else:
            dm = MaskedNodeLightningDataset(train_dataset = train_ds, 
                            val_dataset = val_ds, 
                            batch_size=int(cfg.dataset.batch_size), 
                            pct_mask_nodes=cfg.dataset.pct_mask_nodes,
                            shuffle=True)
        print(f'train ds: {len(train_ds)}, val ds: {len(val_ds)}')
        datasets = [train_ds, val_ds]
        names = ["training", "validation"]

    ####### TRAINING #######
    # Model Initialization
    try:
        if cfg.model.model_type == 'gnn-transformer':
            print('Load GNNTransfomer...')
            if cfg.dataset.pct_mask_nodes > 0:
                model = LitGNNTransformerMasked(cfg)
            else:
                model = LitGNNTransformer(cfg)
        elif cfg.model.model_type == 'gnn':
            print('Load GNN...')
            if cfg.dataset.pct_mask_nodes > 0:
                model = LitGCNMasked(cfg, class_weigths)
            else:
                model = LitGCN(cfg, class_weigths)
        elif cfg.model.model_type == 'fcnn':
            print('Load FCNN...')
            model = BaselineFCNN(cfg)
        elif cfg.model.model_type == 'pca-transformer':
            print('Load PCA Transformer...')
            model = LitPCATransformerMasked(cfg, class_weigths)
        
        # Log total number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if cfg.wandb.use:
            wandb.log({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params
            })
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
    except ValueError:
        print("No valid model defined in .yaml file.")

    steps_per_epoch = math.ceil(len(train_ds) / cfg.dataset.batch_size)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    if 'classification' in cfg.dataset.prediction_task:
        early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.05, patience=10*steps_per_epoch, verbose=False, mode="max")
    if 'regression' in cfg.dataset.prediction_task:
        early_stop_callback = EarlyStopping(monitor="val_r2", min_delta=0.05, patience=10*steps_per_epoch, verbose=False, mode="max")

    data_name = f"{cfg.dataset.name}_{cfg.dataset.prediction_obs}_{cfg.dataset.library_key[-1]}_{len(cfg.dataset.library_key)}_{cfg.optim.seed}"
    run_name = f"{data_name}_{cfg.model.model_type}"

    if cfg.wandb.use:
        wandb.config = cfg
        print('Wandb initialize...')
        # run = wandb.init(project=cfg.wandb.project_name, 
        #                  config=cfg, 
        #                  name=run_name, 
        #                  job_type = 'model_training')
        wandb_logger = WandbLogger(name = run_name, log_model=True) #save at the end of the training
        if 'classification' in cfg.dataset.prediction_task:
            checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max", filename=run_name) # save model if validation accuracy increases
        if 'regression' in cfg.dataset.prediction_task:
            checkpoint_callback = ModelCheckpoint(monitor="val_r2", mode="max", filename=run_name) 
        print('Training...')
        trainer = pl.Trainer(min_epochs=1, 
                         max_epochs=int(cfg.model.n_epochs), 
                         logger=wandb_logger, 
                         enable_progress_bar=False, 
                         callbacks=[lr_monitor, checkpoint_callback, early_stop_callback],
                         log_every_n_steps=steps_per_epoch,
                         #val_check_interval=0.25,  # Validate every 25% of an epoch
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
                         #val_check_interval=0.25,  # Validate every 25% of an epoch
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
            artifact = wandb.Artifact(name=f"{run_name}_model_{cfg.optim.seed}", type="model")
            artifact.add_file(model_checkpoint_path, name=f"{run_name}.ckpt")

            # Log the artifact
            run.log_artifact(artifact)

        run.finish()
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GTLongRange')

    parser.add_argument('--cfg', dest='cfg', type=str, required=True, help='The configuration file path.')
    parser.add_argument('--sweep_cfg', dest='sweep_cfg', type=str, required=True, help='The sweep configuration file path.')
    parser.add_argument('--model_type', dest='model_type', type=str, required=True)
    parser.add_argument('--sweep_goal', dest='sweep_goal', type=str, required=True, help='Choose sweep goal: (1) hyperparameter or (2) robustness.')
    parser.add_argument('--prediction_task', dest='prediction_task', type=str, required=False, 
                       choices=['regression', 'classification'],
                       help='Type of prediction task (regression or classification)')
    args = parser.parse_args()
    
    # Load both base config and sweep config from yaml
    with open(args.sweep_cfg, 'r') as f:  
        yaml_config = yaml.safe_load(f)
    
    sweep_config = yaml_config['sweep_config']
    
    if args.prediction_task == 'classification':
        sweep_config.update({
            'metric': {
                'name': 'val_acc', 
                'goal': 'maximize'},  # Use 'val_r2' for regression tasks
        })
    elif args.prediction_task == 'regression':
        sweep_config.update({
            'metric': {
                'name': 'val_r2', 
                'goal': 'maximize'},  # Use 'val_r2' for regression tasks
        })
    
    if "transformer" not in args.model_type:
        transformer_keys = [key for key in sweep_config['parameters'] if key.startswith("transformer.")]
        for key in transformer_keys:
            del sweep_config['parameters'][key]

    print(sweep_config)
    
    sweep_id = wandb.sweep(sweep_config, project='InterScale_hyperparameter_sweep')
    
    def train_sweep_function():
        # Pass the sweep run object to main
        main_sweep(args.cfg, args.sweep_goal)
    
    # Run the sweep agent
    wandb.agent(sweep_id, function=train_sweep_function)