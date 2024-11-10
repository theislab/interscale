import torch
from torch_geometric.data import Data, Dataset
import random
from sklearn.model_selection import train_test_split
from geome import transforms, ann2data, iterables
import json
import scanpy as sc
from torch_geometric.data.lightning import LightningDataset
import numpy as np

def split_adata(adata, split_obs: str, val_size: float, test_size: float, seed: int, return_summary: bool = True):
    """
    Split the AnnData object into train, val, and optionally test sets.
    
    Parameters:
    - adata: The AnnData object to split.
    - split_obs: The column in .obs to base the split on (optional).
    - test_size: The proportion of the dataset to include in the test split.
    - val_size: The proportion of the dataset to include in the validation split.
    - seed: Random seed for reproducibility.

    The function adds a new column 'split' to adata.obs with values 'train', 'val', 'test'.
    """
    np.random.seed(seed)
    
    # Initialize the 'split' column with None
    adata.obs['split'] = None
    
    if split_obs is not None:
        unique_groups = adata.obs[split_obs].unique()
        
        # Split unique groups into train and temp (val + test)
        train_groups, temp_groups = train_test_split(unique_groups, test_size=test_size + val_size, random_state=seed)
        
        # Further split temp into val and test
        if test_size > 0:
            relative_val_size = val_size / (test_size + val_size)
            val_groups, test_groups = train_test_split(temp_groups, test_size=1 - relative_val_size, random_state=seed)
        else:
            val_groups = temp_groups
            test_groups = []
        
        # Assign 'train', 'val', 'test' based on groups
        adata.obs.loc[adata.obs[split_obs].isin(train_groups), 'split'] = 'train'
        adata.obs.loc[adata.obs[split_obs].isin(val_groups), 'split'] = 'val'
        if test_groups:
            adata.obs.loc[adata.obs[split_obs].isin(test_groups), 'split'] = 'test'
            
    # Generate summary statistics
    if return_summary:
        summary = {
            'counts': adata.obs['split'].value_counts().to_dict(),
            'groups': {
                'train': list(train_groups),
                'val': list(val_groups),
                'test': list(test_groups) if test_groups else []
            }
        }
        print(summary)
        return adata

    return adata

def prepare_geome_dataset(adata, cfg):
    """
    Loads, preprocesses and transforms the defined .h5ad data to a list of PyG data according to cfg file.
    """
    assert ("classification" in (cfg.dataset.prediction_task)) or ("regression" in (cfg.dataset.prediction_task))
    if 'classification' in cfg.dataset.prediction_task:
        assert str(cfg.dataset.prediction_obs) in adata.obs
    assert isinstance(cfg.dataset.library_key, list)
    assert len(cfg.dataset.library_key) >= 0
    assert all(item in adata.obs for item in cfg.dataset.library_key), "Not all library_keys are in adata.obs_names"

    adj_matrix_loc = "adj_matrix"
    prediction_obs = cfg.dataset.prediction_obs
    category_to_iterate_list = cfg.dataset.library_key
    subset_dict = cfg.dataset.subset_dict

    # initalize object to save train, val and test PyG datas
    datas_train, datas_val, datas_test = list(), list(), list()
    
    for category_to_iterate in category_to_iterate_list:
        cfg.dataset.spatial_neigbors_kwargs.merge_from_list(['library_key', category_to_iterate])
        spatial_neigbors_kwargs = cfg.dataset.spatial_neigbors_kwargs

        one_hot_encode_list = [prediction_obs]

        if 'classification' in cfg.dataset.prediction_task:
            fields = {
                "x": ["X"],
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
                "x": ["X"],
                "edge_index": ["uns/edge_index"],
                "obs_names": ["obs_names"],
            }

            preprocess = transforms.Compose(
                [
                    transforms.Subset(key_value = subset_dict, axis="obs"),
                ]
            )


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

        pyg_train, adata_train = list(a2d(adata[adata.obs['split'] == 'train']))
        pyg_val, adata_val = list(a2d(adata[adata.obs['split'] == 'val']))
        datas_train.extend(pyg_train)
        datas_val.extend(pyg_val)
        if 'test' in np.unique(adata.obs['split']):
            pyg_test, adata_test = list(a2d(adata[adata.obs['split'] == 'test']))
            datas_test.extend(pyg_test)
    
    if 'test' in np.unique(adata.obs['split']):
        print('test')
        datas_test, adata_test = list(a2d(adata[adata.obs['split'] == 'test']))
        return [datas_train, datas_val, datas_test], [adata_train, adata_val, adata_test]

    return [datas_train, datas_val], [adata_train, adata_val]

