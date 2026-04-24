import argparse
import os

import psutil
import scanpy as sc
import squidpy as sq
import wandb
import yaml

import interscale as interscale
from interscale.config import load_config
from interscale.geome_dataloader import GraphAnnDataModule
from interscale.pp.segmentation_noise import apply_segmentation_noise
from interscale.tl import prepare_geome_dataset
from interscale.tl.utils import get_model_filename_prefix


def print_memory_usage(stage=""):
    """Print current memory usage for both CPU and GPU"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"[MEMORY] {stage}: {memory_gb:.2f} GB")

    # Add GPU memory monitoring
    try:
        import torch

        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
            gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
            print(f"[GPU MEMORY] {stage}: Allocated: {gpu_allocated:.2f} GB, Reserved: {gpu_reserved:.2f} GB")
    except ImportError:
        pass


def print_memory_debug():
    """Debug function to see what's consuming memory"""
    try:
        import gc

        # Get object counts by type
        objects = gc.get_objects()
        type_counts = {}
        for obj in objects:
            obj_type = type(obj).__name__
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

        # Print top memory consumers
        print("\n[MEMORY DEBUG] Top object types:")
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        for obj_type, count in sorted_types[:10]:
            print(f"  {obj_type}: {count} objects")

    except (RuntimeError, TypeError, ValueError) as e:
        print(f"Memory debug failed: {e}")


def main_sweep(cfg_path, model_type, sweep_goal):

    print_memory_usage("Start of main_sweep")

    cfg = load_config(cfg_path)

    assert cfg.wandb.use, "Wandb is not enabled in the configuration file. Necessary for sweep."

    local_component = False
    global_component = False

    if model_type == "LocalModel" or model_type == "CombinedModel":
        local_component = True
    if model_type == "GlobalModel" or model_type == "CombinedModel":
        global_component = True

    file_name_prefix = get_model_filename_prefix(cfg, local_component, global_component)

    if cfg.wandb.use:
        print("Wandb initialize...")
        sweep_run = wandb.init(
            project=cfg.wandb.project_name, config=cfg, name=file_name_prefix, job_type="model_training"
        )
        sweep_config = wandb.config

    # Update configuration with sweep parameters
    if sweep_config is not None:
        cfg.set_new_allowed(True)
        cfg.defrost()
        print("sweep config: ", sweep_config)
        print("sweep run: ", sweep_run.config)
        if sweep_goal == "robustness":
            print("robustness sweep")
            cfg.dataset.pct_mask_nodes = sweep_config["dataset.pct_mask_nodes"]
            cfg.dataset.spatial_neigbors_kwargs.radius = sweep_config["dataset.spatial_neigbors_kwargs.radius"]
            cfg.optim.seed = sweep_config["optim.seed"]
        elif sweep_goal == "segmentation":
            print("segmentation sweep")
            cfg.dataset.segmentation_robustness = sweep_config["dataset.segmentation_robustness"]
            cfg.optim.seed = sweep_config["optim.seed"]
        elif sweep_goal == "hyperparmeter":
            print("hyperparameter sweep")
            cfg.optim.lr = sweep_config["optim.lr"]
            cfg.optim.lr_warmup = sweep_config["optim.lr_warmup"]
            cfg.optim.wd = sweep_config["optim.wd"]
            cfg.dataset.batch_size = sweep_config["dataset.batch_size"]
            cfg.dataset.pct_mask_nodes = sweep_config["dataset.pct_mask_nodes"]
            cfg.model.n_embed = sweep_config["model.n_embed"]
            if model_type == "LocalModel" or model_type == "CombinedModel":
                print("LocalModel configs")
                cfg.model.local_component.parameters.num_layers = sweep_config[
                    "model.local_component.parameters.num_layers"
                ]
                cfg.model.local_component.parameters.hidden_dim = sweep_config[
                    "model.local_component.parameters.hidden_dim"
                ]
            elif model_type == "GlobalModel" or model_type == "CombinedModel":
                print("transformer configs")
                cfg.model.global_component.parameters.dim_feedforward = sweep_config[
                    "model.global_component.parameters.dim_feedforward"
                ]
                cfg.model.global_component.parameters.num_layers = sweep_config[
                    "model.global_component.parameters.num_layers"
                ]
                cfg.model.global_component.parameters.n_heads = sweep_config[
                    "model.global_component.parameters.n_heads"
                ]
                cfg.model.global_component.parameters.dropout = sweep_config[
                    "model.global_component.parameters.dropout"
                ]
                # cfg.transformer.max_seq_len = sweep_run.config.transformer.max_seq_len
        elif sweep_goal == "loss":
            print("loss sweep")
            cfg.optim.loss = sweep_config["optim.loss"]
        cfg.freeze()

    ####### PREPROCESSING #######
    # Load adata
    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    print_memory_usage("After loading h5ad")
    print(adata)
    if cfg.dataset.segmentation_robustness is not None:
        print("Applying segmentation noise...")
        sq.gr.spatial_neighbors(adata, **cfg.dataset.spatial_neigbors_kwargs)
        adata = apply_segmentation_noise(adata, cfg.dataset.segmentation_robustness)

    if model_type == "LocalModel":
        print_memory_usage("Before LocalModel setup")
        interscale.model.LocalModel._setup_anndata(
            adata=adata,
            prediction_task=cfg.dataset.prediction_task,
            layer_key=cfg.dataset.layer_key,
            sample_key_list=cfg.dataset.sample_key,
            prediction_obs=cfg.dataset.prediction_obs,
        )
        print_memory_usage("After LocalModel setup")

        model = interscale.model.LocalModel(adata, cfg=cfg)
        print_memory_usage("After LocalModel creation")
    elif model_type == "GlobalModel":
        print_memory_usage("Before GlobalModel setup")
        interscale.model.GlobalModel._setup_anndata(
            adata=adata,
            prediction_task=cfg.dataset.prediction_task,
            layer_key=cfg.dataset.layer_key,
            sample_key_list=cfg.dataset.sample_key,
            prediction_obs=cfg.dataset.prediction_obs,
        )
        print_memory_usage("After GlobalModel setup")

        model = interscale.model.GlobalModel(adata, cfg=cfg)
        print_memory_usage("After GlobalModel creation")
    elif model_type == "CombinedModel":
        print_memory_usage("Before CombinedModel setup")
        interscale.model.CombinedModel._setup_anndata(
            adata=adata,
            prediction_task=cfg.dataset.prediction_task,
            layer_key=cfg.dataset.layer_key,
            sample_key_list=cfg.dataset.sample_key,
            prediction_obs=cfg.dataset.prediction_obs,
        )
        print_memory_usage("After CombinedModel setup")

        model = interscale.model.CombinedModel(adata, cfg=cfg)
        print_memory_usage("After CombinedModel creation")

    print_memory_usage("Before prepare_geome_dataset")
    pyg_data_list, _ = prepare_geome_dataset(adata, cfg)
    print_memory_usage("After prepare_geome_dataset")

    dm = GraphAnnDataModule(
        datas=pyg_data_list,
        num_workers=1,
        batch_size=int(cfg.dataset.batch_size),
        pct_mask_nodes=cfg.dataset.pct_mask_nodes,
        learning_type="node",
    )
    print_memory_usage("After datamodule creation")

    model.train(max_epochs=cfg.optim.n_epochs, datamodule=dm, early_stopping=cfg.optim.early_stopping)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GTLongRange")

    parser.add_argument("--cfg", dest="cfg", type=str, required=True, help="The configuration file path.")
    parser.add_argument(
        "--sweep_cfg", dest="sweep_cfg", type=str, required=True, help="The sweep configuration file path."
    )
    parser.add_argument("--model_type", dest="model_type", type=str, required=True)
    parser.add_argument(
        "--sweep_goal",
        dest="sweep_goal",
        type=str,
        required=True,
        help="Choose sweep goal: (1) hyperparameter or (2) robustness.",
    )
    parser.add_argument(
        "--prediction_task",
        dest="prediction_task",
        type=str,
        required=False,
        choices=["regression", "classification"],
        help="Type of prediction task (regression or classification)",
    )
    args = parser.parse_args()

    # Load both base config and sweep config from yaml
    with open(args.sweep_cfg) as f:
        yaml_config = yaml.safe_load(f)

    sweep_config = yaml_config["sweep_config"]

    if args.prediction_task == "classification":
        sweep_config.update(
            {
                "metric": {"name": "val_acc", "goal": "maximize"},
            }
        )
    elif args.prediction_task == "regression":
        sweep_config.update(
            {
                "metric": {"name": "val_r2", "goal": "maximize"},  # Use 'val_r2' for regression tasks
            }
        )

    if "GlobalModel" not in args.model_type or "CombinedModel" not in args.model_type:
        transformer_keys = [key for key in sweep_config["parameters"] if key.startswith("transformer.")]
        for key in transformer_keys:
            del sweep_config["parameters"][key]

    print(sweep_config)

    sweep_id = wandb.sweep(sweep_config, project="InterScale_hyperparameter_sweep")

    def train_sweep_function():
        # Pass the sweep run object to main
        main_sweep(args.cfg, args.model_type, args.sweep_goal)

    # Run the sweep agent
    wandb.agent(sweep_id, function=train_sweep_function)
