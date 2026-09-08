from yacs.config import CfgNode as CN


def get_dataset_cfg(cfg):
    """
    prediction_task: str = [graph, node_classification, node_regression]
    prediction_obs: str = value to be predicted during training, must be in adata.obs
    subset_dict: {adata}
    sample_key: list of keys in adata.obs to split the data into PyG Data objects (e.i. sliding_window, FOV, sample etc)
    num_features: number of gene expressions (added in prepare_geome_function)
    num_features: number of classes in prediction_obs (added in prepare_geome_function)
    mask_strategy: granularity of the reconstruction corruption, "node" or "gene"
    mask_percentage: Bernoulli masking probability -- per cell under mask_strategy "node",
        per (cell, gene) entry under "gene"
    """
    cfg.dataset = CN()

    cfg.dataset.h5ad_data = ""
    cfg.dataset.name = ""
    cfg.dataset.description = ""
    cfg.dataset.prediction_task = "regression"
    cfg.dataset.prediction_obs = None
    cfg.dataset.prediction_level = "node"
    cfg.dataset.layer_key = None  # default: .X
    cfg.dataset.sample_key = []
    cfg.dataset.split_key = "split"

    cfg.dataset.batch_size = 32
    cfg.dataset.train_size = 0.7
    cfg.dataset.val_size = 0.2
    cfg.dataset.test_size = 0.1
    cfg.dataset.num_features = -1
    cfg.dataset.num_classes = -1

    # Reconstruction corruption. "node" blanks whole cells and scores all G genes of them;
    # "gene" blanks individual (cell, gene) entries in every cell and scores those entries only.
    # See interscale.tl.masking for why the two objectives behave so differently -- under "node"
    # the target cell contributes nothing about itself, so the population mean is already a
    # strong solution. "node" stays the default so every existing config is unchanged.
    cfg.dataset.mask_strategy = "node"
    # One rate, not one per strategy: mask_strategy already says what a unit is, so a second key
    # would only ever be the inert half of the pair -- and setting the wrong one silently gives a
    # run with no masking. Note the two strategies are not comparable at equal values: a per-entry
    # rate is a different quantity from a per-cell one (MAE/GraphMAE use 0.25-0.75 for features).
    cfg.dataset.mask_percentage = 0.2

    # Segmentation robustness parameters
    cfg.dataset.segmentation_robustness = None  # [node_fraction, overflow_fraction] or None
    # only needed for segmentation robustness experiments
    cfg.dataset.spatial_neigbors_kwargs = CN()
    cfg.dataset.spatial_neigbors_kwargs.radius = 50
    cfg.dataset.spatial_neigbors_kwargs.coord_type = "generic"
    cfg.dataset.spatial_neigbors_kwargs.library_key = ""
    cfg.dataset.spatial_neigbors_kwargs.n_neighs = 6

    return cfg
