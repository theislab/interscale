from geome import transforms, ann2data, iterables
import numpy as np
from yacs.config import CfgNode as CN

def prepare_a2d_dataset(cfg: CN):
    """
    """
    adj_matrix_loc = "adj_matrix"
    
    category_to_iterate_list = cfg.dataset.sample_key
    prediction_obs = cfg.dataset.prediction_obs
    layer_key = cfg.dataset.layer_key
    
    for category_to_iterate in category_to_iterate_list:
        cfg.dataset.spatial_neigbors_kwargs.merge_from_list(['library_key', category_to_iterate])
        spatial_neigbors_kwargs = cfg.dataset.spatial_neigbors_kwargs

        one_hot_encode_list = [prediction_obs]
        
        X_key = f"layers/{layer_key}" if layer_key is not None else "X"
        print(f"Load GEX from .{X_key}")
        obsm_key = None
        if cfg.model.global_component.parameters.type_gex_embedding == "Precomputed":
            obsm_key = cfg.model.global_component.latent_obsm_key
        if 'classification' in cfg.dataset.prediction_task:
            fields = {
                "x": [X_key],
                "y": [f"obs/{prediction_obs}"],
                "edge_index": ["uns/edge_index"],
                "obs_names": ["obs_names"],
                "sample_key": [f"obs/{category_to_iterate}"],
            }

            preprocess = transforms.Compose(
                [
                    transforms.SaveOneHotEncodeLabels(keys = one_hot_encode_list, axis = 'obs', key_added = 'one_hot')
                ]
            )
        elif 'regression' in cfg.dataset.prediction_task:
            fields = {
                "x": [X_key],
                "edge_index": ["uns/edge_index"],
                "obs_names": ["obs_names"],
                "sample_key": [f"obs/{category_to_iterate}"],
            }
            
            preprocess = None
        if obsm_key is not None:
            fields["embeddings"] = [f"obsm/{obsm_key}"]

        transform = transforms.Compose(
            [
                transforms.AddEdgeIndex(edge_index_key="edge_index", func_args=spatial_neigbors_kwargs, spatial_key="spatial", key_added=adj_matrix_loc),
            ]
        )

        return ann2data.Ann2DataBasic(
            fields=fields,
            adata2iter=iterables.ToCategoryIterator(category_to_iterate, axis="obs", preserve_categories = [prediction_obs]),
            preprocess=preprocess,
            transform=transform,
            save_preprocessed_adata = True,
        )

def prepare_geome_dataset(adata, 
                          cfg, 
                          split_key: str = 'split'):
    """
    Loads, preprocesses and transforms the defined .h5ad data to a list of PyG data according to cfg file.
    """
    assert ("classification" in (cfg.dataset.prediction_task)) or ("regression" in (cfg.dataset.prediction_task))
    if 'classification' in cfg.dataset.prediction_task:
        assert str(cfg.dataset.prediction_obs) in adata.obs
    assert isinstance(cfg.dataset.sample_key, list)
    assert len(cfg.dataset.sample_key) >= 0
    assert all(item in adata.obs for item in cfg.dataset.sample_key), "Not all library_keys are in adata.obs_names"
    
    # Check for duplicate observation names
    adata.obs_names = [str(i) for i in range(1, len(adata.obs_names) + 1)] # ensure that no duplicate observation names are present
    assert len(adata.obs_names) == len(adata.obs_names.unique()), f"Duplicate observation names found. Expected {len(adata.obs_names)} unique names but found {len(adata.obs_names.unique())}"

    # Convert sample_key columns to categorical type to avoid numpy.dtypes.Int64DType error
    for key in cfg.dataset.sample_key:
        if key in adata.obs.columns:
            adata.obs[key] = adata.obs[key].astype('category')

    adj_matrix_loc = "adj_matrix"
    prediction_obs = cfg.dataset.prediction_obs
    category_to_iterate_list = cfg.dataset.sample_key
    layer_key = cfg.dataset.layer_key
    subset_dict = {}
    
    if cfg.dataset.split_key in adata.obs.columns:
        print(f'Split key {cfg.dataset.split_key} already exists in adata.obs')
        split_key = cfg.dataset.split_key

    # initalize object to save train, val and test PyG datas
    datas_train, datas_val, datas_test = list(), list(), list()
    
    for category_to_iterate in category_to_iterate_list:

        cfg.dataset.spatial_neigbors_kwargs.merge_from_list(['library_key', category_to_iterate])
        spatial_neigbors_kwargs = cfg.dataset.spatial_neigbors_kwargs

        one_hot_encode_list = [prediction_obs]
        X_key = f"layers/{layer_key}" if layer_key is not None else "X"

        obsm_key = None
        if cfg.model.global_component.parameters.type_gex_embedding == "Precomputed":
            obsm_key = cfg.model.global_component.latent_obsm_key
        if 'classification' in cfg.dataset.prediction_task:
            fields = {
                "x": [X_key],
                "y": [f"obs/{prediction_obs}"],
                "edge_index": ["uns/edge_index"],
                "obs_names": ["obs_names"],
            }

            preprocess = transforms.Compose(
                [
                    transforms.Subset(key_value = subset_dict, axis="obs"),
                    transforms.Categorize(keys=list(subset_dict.keys()) + one_hot_encode_list, axis="obs"),
                    transforms.SaveOneHotEncodeLabels(keys = one_hot_encode_list, axis = 'obs', key_added = 'one_hot')
                ]
            )
        elif 'regression' in cfg.dataset.prediction_task:
            fields = {
                "x": [X_key],
                "edge_index": ["uns/edge_index"],
                "obs_names": ["obs_names"],
            }

            preprocess = transforms.Compose(
                [
                    transforms.Subset(key_value = subset_dict, axis="obs"),
                ]
            )
        if obsm_key is not None:
            if obsm_key not in adata.obsm:
                raise ValueError(f"Precomputed embeddings key '{obsm_key}' not found in adata.obsm")
            fields["embeddings"] = [f"obsm/{obsm_key}"]
        transform = transforms.Compose(
        [
            transforms.AddEdgeIndex(edge_index_key="edge_index", func_args=spatial_neigbors_kwargs, spatial_key="spatial", key_added=adj_matrix_loc),
        ]
        )

        a2d = ann2data.Ann2DataBasic(
            fields=fields,
            adata2iter=iterables.ToCategoryIterator(category_to_iterate, axis="obs", preserve_categories = [prediction_obs]),
            preprocess=preprocess,
            transform=transform,
            save_preprocessed_adata = True,
        )

        pyg_train, adata_train = list(a2d(adata[adata.obs[split_key] == 'train']))
        pyg_val, adata_val = list(a2d(adata[adata.obs[split_key] == 'val']))
        datas_train.extend(pyg_train)
        datas_val.extend(pyg_val)
        if 'test' in np.unique(adata.obs[split_key]):
            pyg_test, adata_test = list(a2d(adata[adata.obs[split_key] == 'test']))
            datas_test.extend(pyg_test)
    
    if 'test' in np.unique(adata.obs[split_key]):
        datas_test, adata_test = list(a2d(adata[adata.obs[split_key] == 'test']))
        return [datas_train, datas_val, datas_test], [adata_train, adata_val, adata_test]

    return [datas_train, datas_val], [adata_train, adata_val]