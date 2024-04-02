from graph_transformer_long_range_niches.tl.load_config import Config  # noqa, register custom modules
from graph_transformer_long_range_niches.model.gnn_transformer import GNNTransformer, LitGNNTransformer
from graph_transformer_long_range_niches._train import train_gnntransformer, train_gnn
from graph_transformer_long_range_niches.tl.datasets import prepare_dataset, prepare_dataset_split
from graph_transformer_long_range_niches.modules.gcn import GCN, LitGCN

# PyTorch Lightning
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger

import argparse
import os
import yaml
import torch
import wandb
from torch_geometric.loader import DataLoader

def main(cfg_path, default_path=None):

    default_path = '/home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/default_configs.yaml'
    cfg = Config(cfg_path, default_path)
    wandb_use = bool(cfg.get('wandb/use'))
    wandb_logger = None

    # load PyTorch Geometric object
    print(f"Load data {cfg.get('dataset/data_name')}...")
    data, slices = torch.load(cfg.get('dataset/data_path'))
    train_size, val_size = float(cfg.get('dataset/train_size')), float(cfg.get('dataset/val_size'))
    datasets = prepare_dataset_split(data, slices, train_size, val_size)
    train_loader = DataLoader([data for data in datasets if data.train_mask], batch_size=int(cfg.get('dataset/batch_size')), shuffle=True)
    val_loader = DataLoader([data for data in datasets if data.val_mask], batch_size=int(cfg.get('dataset/batch_size')), shuffle=False)

    # WandB 
    if wandb_use:
        print('Wandb initialize...')
        wandb.init(project=cfg.get('wandb/project_name'), config=cfg._data, name=cfg.get('wandb/name'))
        wandb_logger = WandbLogger(log_model="all")

    # model initialization
    try:
        if cfg.get('model/model_type') == 'gnn-transformer':
            print('Load GNNTransfomer...')
            model = LitGNNTransformer(cfg)
        if cfg.get('model/model_type') == 'gnn':
            print('Load GNN...')
            model = LitGCN(cfg)
    except ValueError:
        print("No valid model defined in .yaml file.")

    trainer = pl.Trainer(min_epochs=1, max_epochs=int(cfg.get('model/n_epochs')), logger=wandb_logger, log_every_n_steps=1)
    trainer.fit(model, train_loader, val_loader) # Q: Does this automatically run validation??
    trainer.validate(model=model, dataloaders=val_loader)

    if wandb_use:
        wandb.finish()

def main_manual(cfg_path, default_path=None):

    default_path = '/home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/default_configs.yaml'
    cfg = Config(cfg_path, default_path)
    wandb_use = bool(cfg.get('wandb/use'))

    # load PyTorch Geometric object
    print(f"Load data {cfg.get('dataset/data_name')}...")
    data, slices = torch.load(cfg.get('dataset/data_path'))
    train_size, val_size = float(cfg.get('dataset/train_size')), float(cfg.get('dataset/val_size'))
    datasets = prepare_dataset_split(data, slices, train_size, val_size)
    train_loader = DataLoader([data for data in datasets if data.train_mask], batch_size=int(cfg.get('dataset/batch_size')), shuffle=True)
    val_loader = DataLoader([data for data in datasets if data.val_mask], batch_size=int(cfg.get('dataset/batch_size')), shuffle=False)

    # WandB 
    if wandb_use:
        print('Wandb initialize...')
        wandb.init(project=cfg.get('wandb/project_name'), config=cfg._data, name=cfg.get('wandb/name'))

    # model initialization
    try:
        if cfg.get('model/model_type') == 'gnn-transformer':
            print('Load GNNTransfomer...')
            model = GNNTransformer(cfg)
        if cfg.get('model/model_type') == 'gnn':
            print('Load GNN...')
            model = GCN(cfg)
    except ValueError:
        print("No valid model defined in .yaml file.")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get('model/lr')), weight_decay=float(cfg.get('model/wd')))
    criterion = torch.nn.CrossEntropyLoss()
    if cfg.get('model/model_type') == 'gnn-transformer':
        print('Train GNNTransformer...')
        train_gnntransformer(train_loader, model, optimizer, criterion, wandb_use, int(cfg.get('model/n_epochs')))
    if cfg.get('model/model_type') == 'gnn':
        print('Train GNN...')
        train_gnn(train_loader, model, optimizer, criterion, wandb_use, int(cfg.get('model/n_epochs')))
    if wandb_use:
        wandb.finish()

# cfg_he23_cosmx_lung_full = '/home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/he23_cosmx_human_lung_niche.yaml'
# cfg_he23_cosmx_lung_gnn = '/home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/he23_cosmx_human_lung_niche_gnn.yaml'
# cfg_he23_cosmx_lung_0 = '/home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/he23_cosmx_human_lung_0.yaml'
# default_cfg = '/home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/default_configs.yaml'

# main(cfg_he23_cosmx_lung_gnn,default_cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GTLongRange')

    parser.add_argument('--cfg', dest='cfg', type=str, required=True, help='The configuration file path.')
    args = parser.parse_args()

    main(args.cfg)

