from graph_transformer_long_range_niches.tl.load_config import Config  # noqa, register custom modules
from graph_transformer_long_range_niches.model.gnn_transformer import LitGNNTransformer
from graph_transformer_long_range_niches.pp.geome_utils import prepare_geome_dataset
from graph_transformer_long_range_niches.modules.gcn import LitGCN
from graph_transformer_long_range_niches.tl.wandb import log_data
from graph_transformer_long_range_niches.model.baseline import BaselineFCNN

# PyTorch Lightning
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import argparse
import wandb
from torch_geometric.loader import DataLoader
from torch_geometric.data.lightning import LightningDataset
from sklearn.model_selection import train_test_split
import math

def main(cfg_path):

    cfg = Config(cfg_path)
    wandb_use = bool(cfg.get('wandb/use'))
    wandb_logger = None

    # Geome dataloader
    print('Load PyG data...')
    pyg_datas, adata = prepare_geome_dataset(cfg)
    train_size, val_size, test_size = float(cfg.get('dataset/train_size')), float(cfg.get('dataset/val_size')), float(cfg.get('dataset/test_size'))
    train_ds, val_ds = train_test_split(pyg_datas, train_size=train_size, test_size=val_size+test_size, random_state=42)
    if test_size > 0.0:
        val_ds, test_ds = train_test_split(val_ds, train_size=1-test_size, test_size=test_size, random_state=42)
        dm = LightningDataset(train_dataset = train_ds, 
                              val_dataset = val_ds, 
                              test_dataset = test_ds, 
                              batch_size=int(cfg.get('dataset/batch_size')), 
                              shuffle=True)
        print(f'train ds: {len(train_ds)}, val ds: {len(val_ds)}, test ds: {len(test_ds)}')
        datasets = [train_ds, val_ds, test_ds]
        names = ["training", "validation", "test"]
    else:
        dm = LightningDataset(train_dataset = train_ds, 
                              val_dataset = val_ds, 
                              batch_size=int(cfg.get('dataset/batch_size')), 
                              shuffle=True)
        print(f'train ds: {len(train_ds)}, val ds: {len(val_ds)}')
        datasets = [train_ds, val_ds]
        names = ["training", "validation"]

    # WandB 
    if wandb_use:
        print('Wandb initialize...')
        data_name = f"{cfg.get('dataset/name')}_{cfg.get('dataset/prediction_obs')}_{cfg.get('dataset/library_key')}"
        run_name = f"{data_name}_{cfg.get('model/model_type')}"
        run = wandb.init(project=cfg.get('wandb/project_name'), config=cfg._data, name=run_name, job_type = 'model_training')
        log_data(datasets + [adata], names + ['adata'], cfg, run)
        #cfg._data = wandb.config # make sure that what is logged is same as waht is run
        wandb_logger = WandbLogger(name = run_name, log_model=True) #save at the end of the training
        checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max", filename=run_name) # save model if validation accuracy increases

    # model initialization
    try:
        if cfg.get('model/model_type') == 'gnn-transformer':
            print('Load GNNTransfomer...')
            model = LitGNNTransformer(cfg)
        elif cfg.get('model/model_type') == 'gnn':
            print('Load GNN...')
            model = LitGCN(cfg)
        elif cfg.get('model/model_type') == 'fcnn':
            print('Load FCNN...')
            model = BaselineFCNN(cfg)
    except ValueError:
        print("No valid model defined in .yaml file.")

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    #early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=5, verbose=False, mode="max")

    steps_per_epoch = math.ceil(len(train_ds) / cfg.get('dataset/batch_size'))

    trainer = pl.Trainer(min_epochs=1, 
                         max_epochs=int(cfg.get('model/n_epochs')), 
                         logger=wandb_logger, 
                         enable_progress_bar=False, 
                         callbacks=[lr_monitor, checkpoint_callback],
                         log_every_n_steps=steps_per_epoch,
                         # Sanity checks: Debugging model
                         #overfit_batches=1,
                         )

    print('Training...')
    trainer.fit(model, dm) 
    trainer.validate(model, dm)

    

    if wandb_use:
        ## log model artifact
        model_checkpoint_path = checkpoint_callback.best_model_path

        if model_checkpoint_path:
            # Create an artifact
            artifact = wandb.Artifact(name=f"{run_name}_model", type="model")
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

