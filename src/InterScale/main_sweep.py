import InterScale as interscale
from InterScale.tl import prepare_geome_dataset
from InterScale.geome_dataloader import GraphAnnDataModule
from InterScale.config import load_config
from InterScale.tl.utils import get_model_filename_prefix

import argparse
import scanpy as sc
import wandb
import yaml

def main_sweep(cfg_path, model_type, sweep_goal):
    
    cfg = load_config(cfg_path)  
    
    local_component = False
    global_component = False
    
    if model_type == 'LocalModel' or model_type == 'CombinedModel':
        local_component = True
    elif model_type == 'GlobalModel':
        global_component = True

    file_name_prefix = get_model_filename_prefix(cfg, local_component, global_component)

    if cfg.wandb.use:
        print('Wandb initialize...')
        sweep_run = wandb.init(project=cfg.wandb.project_name, 
                         config=cfg, 
                         name=file_name_prefix, 
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
            cfg.optim.lr = sweep_config['optim.lr']
            cfg.optim.n_epochs = sweep_config['optim.n_epochs']
            #cfg.dataset.batch_size = sweep_run.config.dataset.batch_size
            cfg.optim.lr_warm_up = sweep_config['optim.lr_warm_up']
            cfg.optim.weight_decay = sweep_config['optim.weight_decay']
            if model_type == 'LocalModel' or model_type == 'CombinedModel':
                print('gnn configs')
                cfg.model.local_component.parameters.num_layers = sweep_config['model.local_component.parameters.num_layers']
                cfg.model.local_component.parameters.hidden_dim = sweep_config['model.local_component.parameters.hidden_dim']
                cfg.model.local_component.parameters.embed_dim = sweep_config['model.local_component.parameters.embed_dim']
                cfg.model.local_component.parameters.dropout = sweep_config['model.local_component.parameters.dropout']
            elif model_type == 'GlobalModel' or model_type == 'CombinedModel':
                print('transformer configs')
                cfg.model.global_component.parameters.d_model = sweep_config['model.n_embed'] # input transformer dimension equal to gnn embed dim
                cfg.model.global_component.parameters.dim_feedforward = sweep_config['model.global_component.parameters.dim_feedforward']
                cfg.model.global_component.parameters.num_layers = sweep_config['model.global_component.parameters.num_layers']
                cfg.model.global_component.parameters.n_heads = sweep_config['model.global_component.parameters.n_heads']
                cfg.model.global_component.parameters.dropout = sweep_config['model.global_component.parameters.dropout']
                #cfg.transformer.max_seq_len = sweep_run.config.transformer.max_seq_len
            elif sweep_goal == 'loss':
                print('loss sweep')
                cfg.optim.loss = sweep_config['optim.loss']
        cfg.freeze()

        
    ####### PREPROCESSING #######
    # Load adata
    cfg = load_config(cfg_path)
    print(cfg)
    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    print(adata)
    
    if model_type == "LocalModel":
        interscale.model.LocalModel._setup_anndata(adata = adata,
                                                prediction_task = cfg.dataset.prediction_task, 
                                                layer_key = cfg.dataset.layer_key, 
                                                sample_key_list = cfg.dataset.sample_key, 
                                                prediction_obs = cfg.dataset.prediction_obs, 
                                                group_key = cfg.dataset.group_label)
        
        model = interscale.model.LocalModel(
            adata,
            cfg = cfg
        )
    elif model_type == "GlobalModel":
        interscale.model.GlobalModel._setup_anndata(adata = adata,
                                                prediction_task = cfg.dataset.prediction_task, 
                                                layer_key = cfg.dataset.layer_key, 
                                                sample_key_list = cfg.dataset.sample_key, 
                                                prediction_obs = cfg.dataset.prediction_obs, 
                                                group_key = cfg.dataset.group_label)
        
        model = interscale.model.GlobalModel(
            adata,
            cfg = cfg
        )
    elif model_type == "CombinedModel":
        interscale.model.CombinedModel._setup_anndata(adata = adata,
                                                prediction_task = cfg.dataset.prediction_task, 
                                                layer_key = cfg.dataset.layer_key, 
                                                sample_key_list = cfg.dataset.sample_key, 
                                                prediction_obs = cfg.dataset.prediction_obs, 
                                                group_key = cfg.dataset.group_label)
        
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
    
    model.train(max_epochs = cfg.optim.n_epochs, 
                datamodule = dm,
                early_stopping = True)

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
                'goal': 'maximize'},  
        })
    elif args.prediction_task == 'regression':
        sweep_config.update({
            'metric': {
                'name': 'val_r2', 
                'goal': 'maximize'},  # Use 'val_r2' for regression tasks
        })
    
    if "GlobalModel" not in args.model_type or "CombinedModel" not in args.model_type:
        transformer_keys = [key for key in sweep_config['parameters'] if key.startswith("transformer.")]
        for key in transformer_keys:
            del sweep_config['parameters'][key]

    print(sweep_config)
    
    sweep_id = wandb.sweep(sweep_config, project='InterScale_hyperparameter_sweep')
    
    def train_sweep_function():
        # Pass the sweep run object to main
        main_sweep(args.cfg, args.model_type, args.sweep_goal)
    
    # Run the sweep agent
    wandb.agent(sweep_id, function=train_sweep_function)