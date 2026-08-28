import argparse
import gc
import os
import traceback
import warnings

import psutil
import scanpy as sc
import squidpy as sq
import wandb
import yaml

import interscale as interscale
from interscale.config import load_config
from interscale.config.cli import add_config_args, print_registry, resolve_cfg_from_args
from interscale.config.sweep import ARM_PARAM, apply_sweep_config, build_sweep_config, load_arms
from interscale.geome_dataloader import GraphAnnDataModule
from interscale.pp.segmentation_noise import apply_segmentation_noise
from interscale.tl import prepare_geome_dataset, remove_zero_expression_cells
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


def main_sweep(cfg_factory, model_type, sweep_goal, sweep_params=None, arms=None):
    """Run one sweep trial.

    Parameters
    ----------
    cfg_factory : callable or str or pathlib.Path
        Called with no arguments to build a **fresh** config for this trial. It must be a
        factory, not a config: ``wandb.agent`` invokes this function once per trial, and
        applying a trial's values mutates the config, so a shared object would accumulate
        every previous trial's settings. A path is accepted and wrapped for convenience.
    model_type : str
        One of ``LocalModel``, ``GlobalModel``, ``CombinedModel``.
    sweep_goal : str
        One of ``interscale.config.sweep.SWEEP_GOALS``.
    sweep_params : list, optional
        The parameter names the sweep declares, used to assert that every one of them was
        actually applied to the config.
    arms : dict, optional
        The sweep yaml's ``arms:`` block. Each trial's ``arm`` value expands into that arm's
        coupled dotted overrides.
    """

    print_memory_usage("Start of main_sweep")

    if callable(cfg_factory):
        cfg = cfg_factory()
    else:
        cfg = load_config(cfg_factory)

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
        print("sweep config: ", sweep_config)
        print("sweep run: ", sweep_run.config)
        # Applies whatever dotted keys the sweep declares, and raises if any of them does not
        # exist in the config rather than letting it vary between trials with no effect.
        cfg, _applied = apply_sweep_config(
            cfg,
            sweep_goal,
            sweep_config,
            model_type=model_type,
            sweep_params=sweep_params,
            arms=arms,
        )

        # The name above was derived from the config BEFORE the trial was applied, so every run
        # of a sweep that varies dataset.name or optim.seed -- both of which feed the prefix, and
        # through it the checkpoint filename -- appeared under one identical name. Renaming here
        # makes a run's name the name of the checkpoint it wrote, which is what an analysis
        # loading models back out of the sweep has to match on.
        file_name_prefix = get_model_filename_prefix(cfg, local_component, global_component)
        if sweep_run.name != file_name_prefix:
            print(f"renaming run: {sweep_run.name} -> {file_name_prefix}")
            sweep_run.name = file_name_prefix
        # Recorded as plain summary keys so the wandb API can be filtered on them without
        # re-deriving anything from the nested config blob.
        sweep_run.summary["checkpoint_prefix"] = file_name_prefix
        sweep_run.summary["resolved_dataset_name"] = cfg.dataset.name
        sweep_run.summary["resolved_sample_key"] = list(cfg.dataset.sample_key)
        sweep_run.summary["resolved_seed"] = cfg.optim.seed
        # `parameters` is only added to the schema for a model type that HAS a global component,
        # so a LocalModel sweep must not touch it.
        if global_component:
            sweep_run.summary["resolved_max_seq_len"] = cfg.model.global_component.parameters.max_seq_len
        sweep_run.summary["resolved_model_save"] = cfg.model.save

    ####### PREPROCESSING #######
    # Load adata
    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    print_memory_usage("After loading h5ad")
    adata = remove_zero_expression_cells(adata)
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

    # wandb.agent runs every trial of a sweep inside ONE process, so anything still holding CUDA
    # tensors when a trial ends stays resident and the next trial starts with less GPU memory
    # than the last. Attention memory here is multiplicative in batch x heads x layers over a
    # 4318-long sequence, so one wide trial is enough to fill a 20 GB card -- and without this
    # release every later trial dies of OOM even at 20 MiB allocations, which is exactly how a
    # 60-trial sweep returned one usable result.
    #
    # try/finally, not a trailing statement: the trials that most need the release are precisely
    # the ones that raise (an OOM leaves a half-built model and its activations referenced), and
    # a cleanup placed after the call is skipped exactly then. Without this, one OOM trial
    # poisons every trial after it and the sweep cannot be run near the memory ceiling at all.
    # The failure is caught and re-raised as a NEW exception carrying only the message, because
    # an exception's __traceback__ keeps every frame in the call stack alive, and those frames
    # hold the activations that caused the OOM in the first place. wandb.agent stores the
    # exception it catches, so the original traceback -- and through it the whole trainer.fit()
    # stack -- outlives the trial. Deleting the locals below is then useless: measured on CosMx,
    # GPU memory after "cleanup" went 13.0 -> 15.4 -> 16.1 -> 17.9 -> 18.1 GB across six trials
    # and every one of them OOMed. Binding the error to a plain string lets Python's implicit
    # `del exc` at the end of the except block drop the traceback before gc.collect() runs.
    trial_error = None
    try:
        model.train(max_epochs=cfg.optim.n_epochs, datamodule=dm, early_stopping=cfg.optim.early_stopping)
    except Exception as exc:
        # format_exc() renders the stack to a STRING, so the full traceback survives in the log
        # while no frame (and so no tensor) stays referenced. Keeping only str(exc) made the
        # first CosMx OOM undiagnosable: the allocation turned out to be in the metric
        # collection, not in attention, and nothing in the message said so.
        trial_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        del model, dm, pyg_data_list, adata
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        print_memory_usage("After per-trial cleanup")

    # Re-raised so wandb still records the trial as failed rather than silently succeeding.
    if trial_error is not None:
        raise RuntimeError(f"sweep trial failed: {trial_error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a wandb sweep for a registered (dataset, task) pair.",
        epilog=(
            "examples:\n"
            "  %(prog)s --dataset melton25 --task graph_clas --model_type CombinedModel \\\n"
            "      --sweep_cfg config_files/sweeps/hyperparameters.yaml\n"
            "  %(prog)s --list\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    add_config_args(parser)
    parser.add_argument(
        "--sweep_cfg", dest="sweep_cfg", type=str, required=False, help="The sweep configuration file path."
    )
    parser.add_argument(
        "--model_type",
        dest="model_type",
        type=str,
        default=None,
        choices=["LocalModel", "GlobalModel", "CombinedModel"],
        help="The model type: LocalModel, GlobalModel or CombinedModel.",
    )
    parser.add_argument(
        "--sweep_goal",
        dest="sweep_goal",
        type=str,
        default=None,
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
        "--sweep_id",
        dest="sweep_id",
        type=str,
        default=None,
        help="Join an EXISTING sweep instead of registering a new one. Two uses: several agents "
        "working one sweep in parallel (a slurm array), and resuming a sweep whose agent hit the "
        "walltime -- a grid sweep does not re-run trials it has already completed. The sweep's "
        "own parameter grid is whatever was registered; --sweep_cfg is still required because it "
        "supplies the arms and the parameter names each trial is checked against.",
    )
    parser.add_argument(
        "--create_only",
        dest="create_only",
        action="store_true",
        help="Register the sweep, print its id, and exit without running an agent. Use to create "
        "a sweep on a login node and then submit a slurm array of agents against it with "
        "--sweep_id.",
    )
    parser.add_argument(
        "--prediction_task",
        dest="prediction_task",
        type=str,
        required=False,
        choices=["regression", "classification"],
        help="Type of prediction task. Defaults to the resolved config's dataset.prediction_task.",
    )
    parser.add_argument(
        "--metric",
        dest="metric",
        type=str,
        default=None,
        help="Metric wandb ranks trials by, overriding the prediction task's default "
        "(classification: val_f1_macro, regression: val_r2). Must be a metric the training plan "
        "actually logs, e.g. val_pearson_corr for a regression sweep searching for correlation "
        "rather than calibration. Set optim.monitor to the same metric so checkpoint selection "
        "agrees with what the sweep ranks.",
    )
    parser.add_argument(
        "--metric_goal",
        dest="metric_goal",
        type=str,
        default="maximize",
        choices=["maximize", "minimize"],
        help="Direction for --metric (default: maximize).",
    )
    args = parser.parse_args()

    if args.list_pairs:
        print_registry(args.registry)
        raise SystemExit(0)

    # Not required=True on these, because --list must work without them.
    for flag, value in (
        ("--model_type", args.model_type),
        ("--sweep_goal", args.sweep_goal),
        ("--sweep_cfg", args.sweep_cfg),
    ):
        if value is None:
            parser.error(f"{flag} is required.")

    # Resolve once up front so a bad --dataset/--task fails now rather than after wandb has
    # registered a sweep server-side. Each trial re-resolves its own fresh copy below.
    base_cfg = resolve_cfg_from_args(args, parser=parser)

    # The metric depends on the prediction task, which the resolved config already knows, so the
    # flag is only needed to override it. The per-dataset scripts used to hardcode
    # `--prediction_task classification`, which would rank a regression sweep by val_f1_macro.
    prediction_task = args.prediction_task or base_cfg.dataset.prediction_task

    # Load both base config and sweep config from yaml
    with open(args.sweep_cfg) as f:
        yaml_config = yaml.safe_load(f)

    sweep_config, sweep_params = build_sweep_config(
        yaml_config,
        prediction_task=prediction_task,
        model_type=args.model_type,
        metric=({"name": args.metric, "goal": args.metric_goal} if args.metric else None),
    )
    print(sweep_config)

    arms = load_arms(yaml_config, sweep_config)

    # Fail before wandb.sweep() registers anything if a declared parameter does not exist in this
    # config: the sweep would otherwise burn every trial varying a key that nothing reads.
    #
    # With arms, EVERY arm is checked, not one representative: an arm's overrides are its own set
    # of dotted keys, and a typo in the fourth arm would otherwise only surface once the first
    # three had trained.
    for arm_name in [None] if arms is None else list(sweep_config["parameters"][ARM_PARAM]["values"]):
        trial = dict.fromkeys(sweep_params, None)
        if arm_name is not None:
            trial[ARM_PARAM] = arm_name
        apply_sweep_config(
            base_cfg.clone(),
            args.sweep_goal,
            trial,
            model_type=args.model_type,
            sweep_params=sweep_params,
            arms=arms,
        )

    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"joining existing sweep {args.sweep_project}/{sweep_id}")
    else:
        sweep_id = wandb.sweep(sweep_config, project=args.sweep_project)

    if args.create_only:
        # Printed in a grep-able form so a submit script can capture it.
        print(f"SWEEP_ID={sweep_id}")
        print(f"sweep url: https://wandb.ai/{wandb.Api().default_entity}/{args.sweep_project}/sweeps/{sweep_id}")
        raise SystemExit(0)

    def train_sweep_function():
        # A fresh config per trial: applying a trial's values mutates it, so a shared object
        # would carry the previous trial's settings into this one.
        main_sweep(
            lambda: resolve_cfg_from_args(args, parser=parser),
            args.model_type,
            args.sweep_goal,
            sweep_params=sweep_params,
            arms=arms,
        )

    # Without an explicit count the agent runs until the job's walltime kills it: `random`
    # over this grid has ~1.5M combinations, so it never exhausts them on its own.
    print(f"running {args.count} sweep trials (sweep_id={sweep_id})")
    # project= is required when joining a sweep by bare id: without it the agent looks the sweep
    # up in the default project rather than the one --sweep_project names.
    wandb.agent(sweep_id, function=train_sweep_function, count=args.count, project=args.sweep_project)
