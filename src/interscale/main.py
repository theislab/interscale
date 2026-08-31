import argparse
import warnings
from pathlib import Path

import scanpy as sc
import squidpy as sq

import interscale as interscale
from interscale.config import load_config
from interscale.config.cli import add_config_args, print_registry, resolve_cfg_from_args
from interscale.geome_dataloader import GraphAnnDataModule
from interscale.pp import apply_segmentation_noise
from interscale.tl import prepare_geome_dataset, remove_zero_expression_cells, set_full_reproducibility

# geome calls the deprecated `sq.gr.spatial_neighbors` entrypoint; silence its
# FutureWarnings so they don't flood the training logs.
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*spatial_neighbors.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*n_neighs.*")


def main(cfg, model_type):
    """Train a single model.

    Parameters
    ----------
    cfg : CN or str or pathlib.Path
        An already-resolved config, or a path to a single config file to load.
    model_type : str
        One of ``LocalModel``, ``GlobalModel``, ``CombinedModel``.
    """
    # Accepts a path as well as a cfg so the historical main(cfg_path, model_type) call still
    # works; the CLI now resolves the config itself, since --dataset/--task layers several files.
    if isinstance(cfg, str | Path):
        cfg = load_config(cfg)

    set_full_reproducibility(cfg.optim.seed)
    print(cfg)
    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    adata = remove_zero_expression_cells(adata)
    print(adata)

    if cfg.dataset.segmentation_robustness is not None:
        node_fraction = cfg.dataset.segmentation_robustness[0]
        overflow_fraction = cfg.dataset.segmentation_robustness[1]
        print("\nApplying segmentation noise:")
        print(f"- Node fraction: {node_fraction}")
        print(f"- Overflow fraction: {overflow_fraction}")
        sq.gr.spatial_neighbors(adata, **cfg.dataset.spatial_neigbors_kwargs)
        adata = apply_segmentation_noise(adata, node_fraction, overflow_fraction)

    if model_type == "LocalModel":
        interscale.model.LocalModel._setup_anndata(
            adata=adata,
            prediction_task=cfg.dataset.prediction_task,
            layer_key=cfg.dataset.layer_key,
            sample_key_list=cfg.dataset.sample_key,
            prediction_obs=cfg.dataset.prediction_obs,
        )

        model = interscale.model.LocalModel(adata, cfg=cfg)
    elif model_type == "GlobalModel":
        interscale.model.GlobalModel._setup_anndata(
            adata=adata,
            prediction_task=cfg.dataset.prediction_task,
            layer_key=cfg.dataset.layer_key,
            sample_key_list=cfg.dataset.sample_key,
            prediction_obs=cfg.dataset.prediction_obs,
        )

        model = interscale.model.GlobalModel(adata, cfg=cfg)
    elif model_type == "CombinedModel":
        interscale.model.CombinedModel._setup_anndata(
            adata=adata,
            prediction_task=cfg.dataset.prediction_task,
            layer_key=cfg.dataset.layer_key,
            sample_key_list=cfg.dataset.sample_key,
            prediction_obs=cfg.dataset.prediction_obs,
        )

        model = interscale.model.CombinedModel(adata, cfg=cfg)

    pyg_data_list, _ = prepare_geome_dataset(adata, cfg)
    dm = GraphAnnDataModule(
        datas=pyg_data_list,
        num_workers=1,
        batch_size=int(cfg.dataset.batch_size),
        mask_percentage=cfg.dataset.mask_percentage,
        mask_strategy=cfg.dataset.mask_strategy,
        learning_type=cfg.dataset.prediction_level,
    )

    model.train(max_epochs=cfg.optim.n_epochs, datamodule=dm, early_stopping=cfg.optim.early_stopping)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train one InterScale model on a registered (dataset, task) pair.",
        epilog=(
            "examples:\n"
            "  %(prog)s --dataset melton25 --task graph_clas --model_type CombinedModel\n"
            "  %(prog)s --list\n"
            "  %(prog)s --cfg config_files/legnini_example.yaml --model_type CombinedModel\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    add_config_args(parser)
    parser.add_argument(
        "--model_type",
        dest="model_type",
        type=str,
        default=None,
        choices=["LocalModel", "GlobalModel", "CombinedModel"],
        help="The model type: LocalModel, GlobalModel or CombinedModel.",
    )
    args = parser.parse_args()

    if args.list_pairs:
        print_registry(args.registry)
        raise SystemExit(0)

    # Not required=True because --list must work without it.
    if args.model_type is None:
        parser.error("--model_type is required (LocalModel, GlobalModel or CombinedModel).")

    main(resolve_cfg_from_args(args, parser=parser), args.model_type)
