from yacs.config import CfgNode as CN

def get_dataset_cfg(cfg):
    """
        prediction_task: str = [graph, node_classification, node_regression]
        prediction_obs: str = value to be predicted during training, must be in adata.obs
        subset_dict: {adata}
        num_features: number of gene expressions (added in prepare_geome_function)
        num_features: number of classes in prediction_obs (added in prepare_geome_function)
        pct_mask_nodes: percentage of single nodes to mask during training in a graph
    """
    cfg.dataset = CN()

    cfg.dataset.h5ad_data = ""
    cfg.dataset.name = ""
    cfg.dataset.description = ""
    cfg.dataset.prediction_task = "" 
    cfg.dataset.prediction_obs = "" 
    cfg.dataset.library_key = []
    cfg.dataset.fine_tuning = []
    cfg.dataset.subset_dict = CN()
    cfg.dataset.obs_split = ""

    cfg.dataset.spatial_neigbors_kwargs = CN()
    cfg.dataset.spatial_neigbors_kwargs.radius = 50
    cfg.dataset.spatial_neigbors_kwargs.coord_type = "generic"
    cfg.dataset.spatial_neigbors_kwargs.library_key = ""

    cfg.dataset.batch_size = 32
    cfg.dataset.train_size = 0.7
    cfg.dataset.val_size = 0.2
    cfg.dataset.test_size = 0.1
    cfg.dataset.num_features = -1
    cfg.dataset.num_classes = -1

    cfg.dataset.k_folds = 0
    cfg.dataset.stratify_group = None
    
    cfg.dataset.pct_mask_nodes = 0.2

    return cfg