import InterScale as interscale
from InterScale.tl import prepare_geome_dataset
from InterScale.geome_dataloader import GraphAnnDataModule
from InterScale.config import load_config

import argparse
import scanpy as sc

def main(cfg_path):

    cfg = load_config(cfg_path)
    print(cfg)
    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    print(adata)
    
    # TODO: Make _set_up_anndata work with multiple sample_keys
    
    interscale.model.LocalModel._setup_anndata(adata = adata,
                                               prediction_task = cfg.dataset.prediction_task, 
                                               layer_key = cfg.dataset.layer_key, 
                                               sample_key = cfg.dataset.sample_key, 
                                               prediction_obs = cfg.dataset.prediction_obs, 
                                               group_key = cfg.dataset.group_label)
    
    model = interscale.model.LocalModel(
        adata,
        cfg = cfg
    )

    pyg_data_list, _ = prepare_geome_dataset(adata, cfg)
    dm = GraphAnnDataModule(datas=pyg_data_list, 
                           num_workers=1, 
                           batch_size=int(cfg.dataset.batch_size), 
                           pct_mask_nodes=cfg.dataset.pct_mask_nodes,
                           learning_type="node")
    
    model.train(max_epochs = cfg.optim.n_epochs, 
                datamodule = dm,
                early_stopping = True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GTLongRange')

    parser.add_argument('--cfg', dest='cfg', type=str, required=True, help='The configuration file path.')
    args = parser.parse_args()

    main(args.cfg)