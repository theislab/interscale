import torch
from torch_geometric.data import Data, Dataset
import random
from sklearn.model_selection import train_test_split
from geome import transforms, ann2data, iterables
import json
import scanpy as sc
from torch_geometric.data.lightning import LightningDataset
import numpy as np
from sklearn.model_selection import KFold

def split_adata(adata, split_obs: str = None, val_size: float = 0.1, test_size: float = 0, seed: int = 40, k_splits: int = 0, return_summary: bool = True, split_key: str = 'split', stratify_groups: str = None):
    """
    Split the AnnData object into train, val, and optionally test sets.
    
    Parameters:
    - adata: The AnnData object to split.
    - split_obs: The column in .obs to base the split on (optional).
    - test_size: The proportion of the dataset to include in the test split.
    - val_size: The proportion of the dataset to include in the validation split.
    - seed: Random seed for reproducibility.

    Return:
    -------
        adata: AnnData
            adata with new .obs column(s) '{split_key}' or '{split_key}_{k}' with values 'train', 'val', 'test'.
    """
    np.random.seed(seed)

    if split_obs is not None:
        split_groups = adata.obs[split_obs].unique()

        if stratify_groups is not None:
            group_to_condition = {
                group: adata[adata.obs[split_obs] == group].obs[stratify_groups].iloc[0]
                for group in split_groups
            }
            stratify_labels = [group_to_condition[group] for group in split_groups]
            print(f"Stratifying by conditions: {np.unique(stratify_labels)}")
        else:
            stratify_labels = None

        if k_splits > 0:
            print("K Fold")
            split_groups = adata.obs[split_obs].unique()
            kf = KFold(n_splits=k_splits, shuffle=True, random_state=seed)
            
            for fold, (train_index, val_index) in enumerate(kf.split(split_groups)):
                split_col = f'{split_key}_{fold + 1}'
                adata.obs[split_col] = None
                
                # Get the group names for train and val
                train_groups = split_groups[train_index]
                val_groups = split_groups[val_index]
                
                # Assign 'train' and 'val' based on the groups
                adata.obs.loc[adata.obs[split_obs].isin(train_groups), split_col] = 'train'
                adata.obs.loc[adata.obs[split_obs].isin(val_groups), split_col] = 'val'
            
                if return_summary:
                    summary = {f'{split_col}_{fold + 1}': {
                                'train_groups': list(train_groups),
                                'val_groups': list(val_groups)
                            }
                    }
                    print(summary)
            
            return adata
        
        if k_splits == 0:
            # Initialize the 'split' column with None
            adata.obs[split_key] = None
            
            # Split unique groups into train and temp (val + test)
            train_groups, temp_groups = train_test_split(split_groups, test_size=test_size + val_size, random_state=seed, stratify=stratify_labels)
        
            # Further split temp into val and test
            if test_size > 0:
                print("Test size > 0")
                if len(temp_groups) < 2:
                    raise ValueError("Not enough groups to create non-empty validation and test sets")
                
                # Get stratification labels for temp groups
                if stratify_labels is not None:
                    temp_stratify = [group_to_condition[group] for group in temp_groups]
                else:
                    temp_stratify = None
                    
                relative_val_size = val_size / (test_size + val_size)
                val_groups, test_groups = train_test_split(
                    temp_groups, 
                    test_size=1 - relative_val_size, 
                    random_state=seed,
                    stratify=temp_stratify
                )
                
                # Verify test set is not empty
                if len(test_groups) == 0:
                    raise ValueError("Test size > 0 but resulted in empty test set")
            else:
                val_groups = temp_groups
                test_groups = []
            
            # Assign 'train', 'val', 'test' based on groups
            adata.obs.loc[adata.obs[split_obs].isin(train_groups), split_key] = 'train'
            adata.obs.loc[adata.obs[split_obs].isin(val_groups), split_key] = 'val'
            if test_groups:
                adata.obs.loc[adata.obs[split_obs].isin(test_groups), split_key] = 'test'
            
            # Generate summary statistics
            if return_summary:
                summary = {
                    'counts': adata.obs[split_key].value_counts().to_dict(),
                    'groups': {
                        'train': list(train_groups),
                        'val': list(val_groups),
                        'test': list(test_groups) if test_groups else []
                    }
                }
                print(summary)
            return adata

    return adata

def prepare_geome_dataset(adata, cfg, split_key: str = 'split'):
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

