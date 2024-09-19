from yacs.config import CfgNode as CN

def get_dataset_cfg(cfg):
    """
        prediction_task: str = [graph, node_classification, node_regression]
        prediction_obs: str = value to be predicted during training, must be in adata.obs
        subset_dict: {adata}
        num_features: number of gene expressions (added in prepare_geome_function)
        num_features: number of classes in prediction_obs (added in prepare_geome_function)
    """
    cfg.dataset = CN()

    cfg.dataset.h5ad_data = ""
    cfg.dataset.name = ""
    cfg.dataset.description = ""
    cfg.dataset.prediction_task = "" 
    cfg.dataset.prediction_obs = "" 
    cfg.dataset.library_key = ""
    cfg.dataset.fine_tuning = []
    cfg.dataset.subset_dict = CN()

    cfg.dataset.spatial_neigbors_kwargs = CN()
    cfg.dataset.spatial_neigbors_kwargs.radius = 50
    cfg.dataset.spatial_neigbors_kwargs.coord_type = "generic"
    cfg.dataset.spatial_neigbors_kwargs.library_key = ""

    cfg.dataset.batch_size = 20
    cfg.dataset.train_size = 0.8
    cfg.dataset.val_size = 0.2
    cfg.dataset.test_size = 0.0
    cfg.dataset.num_features = -1
    cfg.dataset.num_classes = -1

    return cfg