import torch
from torch_geometric.data import Data, Dataset
import random
from sklearn.model_selection import train_test_split
from geome import iterables, transforms, ann2data, datamodule
import json
import scanpy as sc

def prepare_geome_dataset(cfg):
    """
    Loads, preprocesses and transforms the defined .h5ad data to a list of PyG data.
    """
    adj_matrix_loc = "obsp/adj_matrix"
    fields = {
        "x": ["X"],
        "y": [f"obs/{cfg.get('dataset/prediction_obs')}"],
    }
    category_to_iterate = str(cfg.get('dataset/graph_id'))
    #subset_dict = json.loads(cfg.get('dataset/subset_dict')) #ToDo: works with empty subset?
    subset_dict = cfg.get('dataset/subset_dict')
    spatial_neigbors_kwargs = cfg.get('dataset/spatial_neigbors_kwargs')

    preprocess = transforms.Compose(
        [
            transforms.Subset(key_value = subset_dict, axis="obs"), 
            transforms.Categorize(keys=list(subset_dict.keys()) + [cfg.get('dataset/prediction_obs'), cfg.get('dataset/graph_id')], axis="obs"),
            transforms.AddAdjMatrix(adj_matrix_loc, overwrite=True, **spatial_neigbors_kwargs),
            transforms.AddEdgeIndex(adj_matrix_loc, edge_index_key="edge_index", overwrite=True)
        ]
    )

    transform = transforms.Compose(
        [
            transforms.AddDesignMatrix(
                f"obs/{cfg.get('dataset/prediction_obs')}",
                f"obs/{cfg.get('dataset/graph_id')}",
                adj_matrix_loc,
                "design_matrix",
                overwrite=True,
            ),
        ]
    )

    a2d = ann2data.Ann2DataByCategory(
        fields=fields,
        category=category_to_iterate,
        preprocess=preprocess,
        transform=transform,
    )

    adata = sc.read_h5ad(cfg.get('dataset/h5ad_data'))

    datas = list(a2d(adata))
    print(datas[:3])
    return datas


class CustomGraphDataset_graphLabel(Dataset):
    def __init__(self, num_graphs, max_nodes, max_edges, nr_node_features, nr_classes):
        self.num_graphs = num_graphs
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.nr_node_features = nr_node_features
        self.data_list = []

        for _ in range(num_graphs):
            num_nodes = random.randint(1, max_nodes)
            num_edges = random.randint(0, min(num_nodes * (num_nodes - 1) // 2, max_edges))
            
            edge_index = torch.zeros((2, num_edges), dtype=torch.long)
            for i in range(num_edges):
                edge_index[0, i] = random.randint(0, num_nodes - 1)
                edge_index[1, i] = random.randint(0, num_nodes - 1)
            
            x = torch.rand((num_nodes, nr_node_features), dtype=torch.float)  # Node features (random for example)
            y = torch.tensor([random.randint(0, nr_classes)], dtype=torch.long)  # Graph label (binary for example)

            data = Data(x=x, edge_index=edge_index, y=y)
            self.data_list.append(data)

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        return self.data_list[idx]
    
class CustomGraphDataset_nodeLabel(Dataset):
    def __init__(self, num_graphs, max_nodes, max_edges, nr_node_features, nr_classes):
        self.num_graphs = num_graphs
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.nr_node_features = nr_node_features
        self.nr_classes = nr_classes
        self.data_list = []

        for _ in range(num_graphs):
            num_nodes = random.randint(1, max_nodes)
            num_edges = random.randint(0, min(num_nodes * (num_nodes - 1) // 2, max_edges))
            
            edge_index = torch.zeros((2, num_edges), dtype=torch.long)
            for i in range(num_edges):
                edge_index[0, i] = random.randint(0, num_nodes - 1)
                edge_index[1, i] = random.randint(0, num_nodes - 1)
            
            x = torch.rand((num_nodes, nr_node_features), dtype=torch.float)  # Node features (random for example)
            y = torch.randint(0, nr_classes, (num_nodes,), dtype=torch.long)  # Node labels

            data = Data(x=x, edge_index=edge_index, y=y)
            self.data_list.append(data)

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        return self.data_list[idx]
    
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

def prepare_dataset(data, slices):
    """Create a pytorch geometric dataset."""
    datasets = []
    # Iterate over each slice in lung_slices
    for i in range(len(slices['x']) - 1):
        slice_data = {}
        start_index_node = slices['x'][i].item()
        end_index_node = slices['x'][i + 1].item()
        start_index_edge = slices['edge_index'][i].item()
        end_index_edge = slices['edge_index'][i + 1].item()
    
        # Extract relevant data from lung_data for this slice
        slice_data['x'] = data.x[start_index_node:end_index_node]
        slice_data['edge_index'] = data.edge_index[:,start_index_edge:end_index_edge]
        slice_data['edge_attr'] = data.edge_attr[start_index_edge:end_index_edge]
        slice_data['y'] = data.y[start_index_node:end_index_node]
    
        # Create PyTorch Geometric Data object for this slice
        data_slice = Data(**slice_data)
        datasets.append(data_slice)
        
    return datasets
