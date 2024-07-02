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
    Loads, preprocesses and transforms the defined .h5ad data to a list of PyG data.
    """
    adj_matrix_loc = "adj_matrix"
    prediction_obs = cfg.get('dataset/prediction_obs')
    category_to_iterate = str(cfg.get('dataset/library_key'))
    subset_dict = cfg.get('dataset/subset_dict')
    spatial_neigbors_kwargs = cfg.get('dataset/spatial_neigbors_kwargs')
    spatial_neigbors_kwargs['library_key'] = category_to_iterate
    one_hot_encode_list = [prediction_obs]

    fields = {
        "x": ["X"],
        "y": [f"obs/{prediction_obs}"],
        "edge_index": ["uns/edge_index"],
        "obs_names": ["obs_names"],
    }

    if len(cfg.get('dataset/fine_tuning')) > 0:
        for task in cfg.get('dataset/fine_tuning'):
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

    adata = sc.read_h5ad(cfg.get('dataset/h5ad_data'))
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
    cfg.set('dataset/num_features', len(datas[0].x[1]))
    cfg.set('dataset/num_classes', len(datas[0].y[1]))

    # save labels for evaluation
    cfg.set('labels/prediction_obs', np.unique(adata_processed.obs[prediction_obs]))

    print(datas[:3])
    print(len(datas))
    return datas, adata_processed

def load_pyg_data(cfg):
    print('Load PyG data...')
    pyg_datas = prepare_geome_dataset(cfg)
    train_size, val_size, test_size = float(cfg.get('dataset/train_size')), float(cfg.get('dataset/val_size')), float(cfg.get('dataset/test_size'))
    train_ds, val_ds = train_test_split(pyg_datas, train_size=train_size, test_size=val_size+test_size, random_state=42)
    if test_size > 0.0:
        val_ds, test_ds = train_test_split(val_ds, train_size=1-test_size, test_size=test_size, random_state=42)
        dm = LightningDataset(train_dataset = train_ds, 
                              val_dataset = val_ds, 
                              test_dataset = test_ds, 
                              batch_size=int(cfg.get('dataset/batch_size')), 
                              shuffle=True)
        print(f'train ds: {len(train_ds)}, val ds: {len(val_ds)}, test ds: {len(test_ds)}')
        datasets = [train_ds, val_ds, test_ds]
        names = ["training", "validation", "test"]
        
    else:
        dm = LightningDataset(train_dataset = train_ds, 
                              val_dataset = val_ds, 
                              batch_size=int(cfg.get('dataset/batch_size')), 
                              shuffle=True)
        print(f'train ds: {len(train_ds)}, val ds: {len(val_ds)}')
        datasets = [train_ds, val_ds]
        names = ["training", "validation"]
    return datasets, names

    
def prepare_dataset_split(data, slices, train_size=0.8, val_size=0.2):
    """Create a PyTorch Geometric dataset with slice-level train/validation split."""
    datasets = []
    num_slices = len(slices['x']) - 1
    
    # Split slice indices into train and validation sets
    slice_indices = list(range(num_slices))
    train_slices, val_slices = train_test_split(slice_indices, train_size=train_size, test_size=val_size, random_state=42)
    
    # Iterate over each slice in lung_slices
    for i in range(num_slices):
        slice_data = {}
        start_index_node = slices['x'][i].item()
        end_index_node = slices['x'][i + 1].item()
        start_index_edge = slices['edge_index'][i].item()
        end_index_edge = slices['edge_index'][i + 1].item()
    
        # Extract relevant data from lung_data for this slice
        slice_data['x'] = data.x[start_index_node:end_index_node]
        slice_data['edge_index'] = data.edge_index[:, start_index_edge:end_index_edge]
        slice_data['edge_attr'] = data.edge_attr[start_index_edge:end_index_edge]
        slice_data['y'] = data.y[start_index_node:end_index_node]
    
        # Create PyTorch Geometric Data object for this slice
        data_slice = Data(**slice_data)
        
        # Assign slice to train or validation set
        if i in train_slices:
            data_slice.train_mask = torch.tensor(True)  # Mark as train
            data_slice.val_mask = torch.tensor(False)  # Mark as not validation
        elif i in val_slices:
            data_slice.train_mask = torch.tensor(False)  # Mark as not train
            data_slice.val_mask = torch.tensor(True)  # Mark as validation
        else:
            raise ValueError("Invalid slice index")
        
        datasets.append(data_slice)
        
    return datasets
