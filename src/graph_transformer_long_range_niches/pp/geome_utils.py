import torch
from torch_geometric.data import Data, Dataset
import random
from sklearn.model_selection import train_test_split
from geome import transforms, ann2data, iterables
import json
import scanpy as sc
from torch_geometric.data.lightning import LightningDataset
import numpy as np

def prepare_geome_dataset(cfg):
    """
    Loads, preprocesses and transforms the defined .h5ad data to a list of PyG data according to cfg file.
    """
    adj_matrix_loc = "adj_matrix"
    prediction_obs = cfg.dataset.prediction_obs
    category_to_iterate = cfg.dataset.library_key
    subset_dict = cfg.dataset.subset_dict
    cfg.dataset.spatial_neigbors_kwargs.merge_from_list(['library_key', category_to_iterate])
    spatial_neigbors_kwargs = cfg.dataset.spatial_neigbors_kwargs
    one_hot_encode_list = [prediction_obs]

    fields = {
        "x": ["X"],
        "y": [f"obs/{prediction_obs}"],
        "edge_index": ["uns/edge_index"],
        "obs_names": ["obs_names"],
    }

    if len(cfg.dataset.fine_tuning) > 0:
        for task in cfg.dataset.fine_tuning:
            fields.update({f"y_{task}": [f"obs/{task}"]})
            one_hot_encode_list.append(task)

    preprocess = transforms.Compose(
        [
            transforms.Subset(key_value = subset_dict, axis="obs"), 
            transforms.Categorize(keys=list(subset_dict.keys()) + one_hot_encode_list, axis="obs"),
            transforms.SaveOneHotEncodeLabels(keys = one_hot_encode_list, axis = 'obs', key_added = 'one_hot')
        ]
    )

    transform = transforms.Compose(
        
        [
            transforms.AddEdgeIndex(edge_index_key="edge_index", func_args=spatial_neigbors_kwargs, spatial_key="spatial", key_added=adj_matrix_loc),
        ]
    )

    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    adata.obs_names_make_unique()

    a2d = ann2data.Ann2DataBasic(
        fields=fields,
        adata2iter=iterables.ToCategoryIterator(category_to_iterate, axis="obs", preserve_categories = [prediction_obs]),
        preprocess=preprocess,
        transform=transform,
        save_preprocessed_adata = True,
    )

    datas, adata_processed = list(a2d(adata))

    # set number of classes and number of features
    cfg.dataset.merge_from_list(['num_features', len(datas[0].x[1])])
    cfg.dataset.merge_from_list(['num_classes', len(datas[0].y[1])])

    print(datas[:3])
    print(len(datas))
    return datas, adata_processed, cfg


def load_pyg_data(cfg):
    print('Load PyG data...')
    pyg_datas = prepare_geome_dataset(cfg)
    train_size, val_size, test_size = float(cfg.dataset.train_size), float(cfg.dataset.val_size), float(float(cfg.dataset.test_size))
    train_ds, val_ds = train_test_split(pyg_datas, train_size=train_size, test_size=val_size+test_size, random_state=42)
    if test_size > 0.0:
        val_ds, test_ds = train_test_split(val_ds, train_size=1-test_size, test_size=test_size, random_state=42)
        dm = LightningDataset(train_dataset = train_ds, 
                              val_dataset = val_ds, 
                              test_dataset = test_ds, 
                              batch_size=int(cfg.dataset.batch_size), 
                              shuffle=True)
        print(f'train ds: {len(train_ds)}, val ds: {len(val_ds)}, test ds: {len(test_ds)}')
        datasets = [train_ds, val_ds, test_ds]
        names = ["training", "validation", "test"]
        
    else:
        dm = LightningDataset(train_dataset = train_ds, 
                              val_dataset = val_ds, 
                              batch_size=int(cfg.dataset.batch_size), 
                              shuffle=True)
        print(f'train ds: {len(train_ds)}, val ds: {len(val_ds)}')
        datasets = [train_ds, val_ds]
        names = ["training", "validation"]
    return datasets, names
