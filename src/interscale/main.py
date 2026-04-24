import argparse

import scanpy as sc
import squidpy as sq

import interscale as interscale
from interscale.config import load_config
from interscale.geome_dataloader import GraphAnnDataModule
from interscale.pp import apply_segmentation_noise
from interscale.tl import prepare_geome_dataset, remove_zero_expression_cells, set_full_reproducibility


def main(cfg_path, model_type):

    cfg = load_config(cfg_path)
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
        pct_mask_nodes=cfg.dataset.pct_mask_nodes,
        learning_type="node",
    )

    model.train(max_epochs=cfg.optim.n_epochs, datamodule=dm, early_stopping=cfg.optim.early_stopping)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GTLongRange")

    parser.add_argument("--cfg", dest="cfg", type=str, required=True, help="The configuration file path.")
    parser.add_argument(
        "--model_type",
        dest="model_type",
        type=str,
        required=True,
        help="The model type: LocalModel, GlobalModel or CombinedModel.",
    )
    args = parser.parse_args()

    main(args.cfg, args.model_type)
