from yacs.config import CfgNode as CN


def get_dataset_cfg(cfg):
    """
    prediction_task: str = [graph, node_classification, node_regression]
    prediction_obs: str = value to be predicted during training, must be in adata.obs
    subset_dict: {adata}
    sample_key: list of keys in adata.obs to split the data into PyG Data objects (e.i. sliding_window, FOV, sample etc)
    num_features: number of gene expressions (added in prepare_geome_function)
    num_features: number of classes in prediction_obs (added in prepare_geome_function)
    pct_mask_nodes: percentage of single nodes to mask during training in a graph
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

    cfg.dataset.pct_mask_nodes = 0.2

    # Segmentation robustness parameters
    cfg.dataset.segmentation_robustness = None  # [node_fraction, overflow_fraction] or None
    # only needed for segmentation robustness experiments
    cfg.dataset.spatial_neigbors_kwargs = CN()
    cfg.dataset.spatial_neigbors_kwargs.radius = 50
    cfg.dataset.spatial_neigbors_kwargs.coord_type = "generic"
    cfg.dataset.spatial_neigbors_kwargs.library_key = ""
    cfg.dataset.spatial_neigbors_kwargs.n_neighs = 6

    return cfg
