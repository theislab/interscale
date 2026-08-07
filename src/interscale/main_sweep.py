import argparse
import os
import warnings

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

# geome calls the deprecated `sq.gr.spatial_neighbors` entrypoint; silence its
# FutureWarnings so they don't flood the training logs.
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*spatial_neighbors.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*n_neighs.*")


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


def main_sweep(cfg_path, model_type, sweep_goal, sweep_params=None):

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
            applied = []

            def _apply(node, attr, key):
                """Assign a swept value only if the sweep actually declares that key.

                Staged sweeps vary a subset of the parameters (e.g. optimiser only), so an
                unconditional lookup would KeyError on every key the stage omits.
                """
                if key in sweep_config.keys():
                    setattr(node, attr, sweep_config[key])
                    applied.append(key)

            _apply(cfg.optim, "lr", "optim.lr")
            _apply(cfg.optim, "lr_warmup", "optim.lr_warmup")
            _apply(cfg.optim, "wd", "optim.wd")
            _apply(cfg.dataset, "batch_size", "dataset.batch_size")
            _apply(cfg.dataset, "pct_mask_nodes", "dataset.pct_mask_nodes")
            _apply(cfg.model, "n_embed", "model.n_embed")

            # Two separate `if`s, not if/elif: CombinedModel has BOTH components, and an
            # `elif model_type == "GlobalModel" or model_type == "CombinedModel"` is
            # unreachable for CombinedModel, so its transformer was never swept.
            local = cfg.model.local_component.parameters
            glob = cfg.model.global_component.parameters
            if model_type in ("LocalModel", "CombinedModel"):
                print("LocalModel configs")
                _apply(local, "num_layers", "model.local_component.parameters.num_layers")
                _apply(local, "hidden_dim", "model.local_component.parameters.hidden_dim")
                _apply(local, "dropout_local", "model.local_component.parameters.dropout_local")
            if model_type in ("GlobalModel", "CombinedModel"):
                print("transformer configs")
                _apply(glob, "dim_feedforward", "model.global_component.parameters.dim_feedforward")
                _apply(glob, "num_layers", "model.global_component.parameters.num_layers")
                _apply(glob, "n_heads", "model.global_component.parameters.n_heads")
                # The config key is `dropout_global` (see global_component_config.py); assigning
                # to `dropout` silently created a dead key because set_new_allowed(True) is on.
                _apply(glob, "dropout_global", "model.global_component.parameters.dropout_global")

            if sweep_params is not None:
                ignored = sorted(set(sweep_params) - set(applied))
                if ignored:
                    print(
                        f"WARNING: sweep declares parameters that nothing applies, so they vary "
                        f"between trials with no effect: {ignored}"
                    )
            print(f"applied sweep parameters: {sorted(applied)}")
        elif sweep_goal == "loss":
            print("loss sweep")
            cfg.optim.loss = sweep_config["optim.loss"]
        else:
            raise ValueError(
                f"Unknown --sweep_goal '{sweep_goal}'. Must be one of: "
                "robustness, segmentation, hyperparmeter, loss. "
                "(Nothing would be swept otherwise -- every trial would train the base config.)"
            )
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
        learning_type=cfg.dataset.prediction_level,
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
        # Note the spelling of "hyperparmeter" -- it is what main_sweep() matches on. Without
        # choices=, a typo silently trained the unmodified base config on every trial.
        choices=["robustness", "segmentation", "hyperparmeter", "loss"],
        help="Choose sweep goal: robustness, segmentation, hyperparmeter (sic) or loss.",
    )
    parser.add_argument(
        "--count",
        dest="count",
        type=int,
        default=30,
        help="Number of sweep trials this agent runs before exiting. Without a bound the agent "
        "runs until the SLURM walltime kills it.",
    )
    parser.add_argument(
        "--sweep_project",
        dest="sweep_project",
        type=str,
        default="InterScale_hyperparameter_sweep",
        help="wandb project the sweep is registered under.",
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

    # "val_acc" was never a logged metric name -- the classification MetricCollection logs
    # val_accuracy / val_f1_micro / val_f1_macro / val_f1_<class>. val_f1_macro also matches
    # what EarlyStopping and ModelCheckpoint monitor. Only override the yaml when the flag
    # is given, so a sweep config carrying its own metric block still works without it.
    if args.prediction_task == "classification":
        sweep_config["metric"] = {"name": "val_f1_macro", "goal": "maximize"}
    elif args.prediction_task == "regression":
        sweep_config["metric"] = {"name": "val_r2", "goal": "maximize"}

    if "metric" not in sweep_config:
        raise ValueError(
            "Sweep config declares no `metric`, and --prediction_task was not given to supply "
            "one. wandb would have nothing to rank trials by (and method: bayes cannot run)."
        )

    # Drop parameters for components this model_type does not have, so wandb does not sample
    # values that nothing consumes. The previous filter looked for a `transformer.` prefix,
    # which no key has ever used, under a condition that is true for every model_type.
    drop_prefixes = []
    if args.model_type not in ("LocalModel", "CombinedModel"):
        drop_prefixes.append("model.local_component.")
    if args.model_type not in ("GlobalModel", "CombinedModel"):
        drop_prefixes.append("model.global_component.")
    for key in [k for k in sweep_config["parameters"] if any(k.startswith(p) for p in drop_prefixes)]:
        print(f"dropping sweep parameter not used by {args.model_type}: {key}")
        del sweep_config["parameters"][key]

    sweep_params = list(sweep_config["parameters"])
    print(sweep_config)

    sweep_id = wandb.sweep(sweep_config, project=args.sweep_project)

    def train_sweep_function():
        # Pass the sweep run object to main
        main_sweep(args.cfg, args.model_type, args.sweep_goal, sweep_params=sweep_params)

    # Without an explicit count the agent runs until the job's walltime kills it: `random`
    # over this grid has ~1.5M combinations, so it never exhausts them on its own.
    print(f"running {args.count} sweep trials (sweep_id={sweep_id})")
    wandb.agent(sweep_id, function=train_sweep_function, count=args.count)
