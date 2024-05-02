from graph_transformer_long_range_niches.tl.load_config import Config  # noqa, register custom modules
from graph_transformer_long_range_niches.model.gnn_transformer import LitGNNTransformer
from graph_transformer_long_range_niches._train import train_gnntransformer, train_gnn
from graph_transformer_long_range_niches.pp.datasets import prepare_dataset, prepare_dataset_split, prepare_geome_dataset
from graph_transformer_long_range_niches.modules.gcn import LitGCN
from graph_transformer_long_range_niches.pp.datamodule_geome import GraphAnnDataModule

# PyTorch Lightning
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import argparse
import wandb
from torch_geometric.loader import DataLoader
from torch_geometric.data.lightning import LightningDataset
from sklearn.model_selection import train_test_split

def main(cfg_path, default_path=None):

    default_path = '/home/icb/francesca.drummer/1-Projects/GT-long-range-niches/src/config_files/default_configs.yaml'
    cfg = Config(cfg_path, default_path)
    wandb_use = bool(cfg.get('wandb/use'))
    wandb_logger = None

    # Geome dataloader
    print('Load PyG data...')
    pyg_datas = prepare_geome_dataset(cfg)
    print(pyg_datas[:2])
    train_size, val_size, test_size = float(cfg.get('dataset/train_size')), float(cfg.get('dataset/val_size')), float(cfg.get('dataset/test_size'))
    train_ds, val_ds = train_test_split(pyg_datas, train_size=train_size, test_size=val_size+test_size, random_state=42)
    if test_size > 0.0:
        val_ds, test_ds = train_test_split(val_ds, train_size=1-test_size, test_size=test_size, random_state=42)
        dm = LightningDataset(train_ds, val_ds, test_ds, batch_size=int(cfg.get('dataset/batch_size')), shuffle=True)
    else:
        dm = LightningDataset(train_ds, val_ds, batch_size=int(cfg.get('dataset/batch_size')), shuffle=True)

    # load PyTorch Geometric object
    # print(f"Load data {cfg.get('dataset/data_name')}...")
    # data, slices = torch.load(cfg.get('dataset/data_path'))
    # train_size, val_size = float(cfg.get('dataset/train_size')), float(cfg.get('dataset/val_size'))
    # datasets = prepare_dataset_split(data, slices, train_size, val_size)
    # train_loader = DataLoader([data for data in datasets if data.train_mask], batch_size=int(cfg.get('dataset/batch_size')), shuffle=True)
    # val_loader = DataLoader([data for data in datasets if data.val_mask], batch_size=int(cfg.get('dataset/batch_size')), shuffle=False)

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

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(min_epochs=1, max_epochs=int(cfg.get('model/n_epochs')), logger=wandb_logger, enable_progress_bar=False, log_every_n_steps=10,
                         callbacks=[lr_monitor],
                         # Sanity checks: Debugging model
                         overfit_batches=1
                         )
    print('Training...')
    trainer.fit(model, dm) 
    trainer.validate(model, dm)

    if wandb_use:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GTLongRange')

    parser.add_argument('--cfg', dest='cfg', type=str, required=True, help='The configuration file path.')
    args = parser.parse_args()

    main(args.cfg)

